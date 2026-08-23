// AVX512-VNNI dpbusd microbenchmark (plan M1).
//
// Measures, on one pinned core:
//  (1) dpbusd zmm throughput (8 independent chains, register operands)
//  (2) dpbusd zmm latency (single dependent chain)
//  (3) throughput with a per-dpbusd 4B load + vpbroadcastd (the kernel's u4b build)
//  (4) throughput with a per-dpbusd memory-source vpbroadcastd (the proposed fix)
//  (5) kernel per-subblock shape: 8x {64B qv load + 4x {4B load, broadcast, dpbusd}}
//      = 16 dpbusd per subblock, NA=4 chains (q4_K/q5_K band pass)
//  (6) shape (5) + int split-pass correction (mullo/padd/set1 into 2 acc sets)
//
// Output: cycles/op at the core's measured frequency (scaling_cur_freq sampled
// around the timed loop). TF/core = 128 FLOP per dpbusd.
//
// Build: g++ -O3 -mavx512f -mavx512vl -mavx512dq -mavx512vnni microbench-dpbusd.cpp -o microbench-dpbusd
// Run:   taskset -c 8 ./microbench-dpbusd   (repeat on another core/CCD)

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef __AVX512VNNI__
#error "build with -mavx512f -mavx512vl -mavx512dq -mavx512vnni"
#endif

static uint64_t rdtsc(void) {
    uint32_t lo, hi;
    // "memory" clobber: without it GCC can move the register-only compute
    // loops across the timing boundary
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi) : : "memory");
    return ((uint64_t) hi << 32) | lo;
}

// volatile sink: a written-but-unread static lets GCC delete the whole
// timed chain (that is what happened to the register-only loops)
static volatile uint64_t sink_v = 0;
static void sink_write(const __m512i & s) {
    sink_v = (uint64_t) _mm512_reduce_add_epi32(s);
}

static void warmup(void) {
    volatile uint64_t s = 0;
    for (int i = 0; i < (1 << 24); i++) s += i;
    (void) s;
}

static unsigned freq_khz(const char * cpu) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%s/cpufreq/scaling_cur_freq", cpu);
    FILE * f = fopen(path, "r");
    if (!f) return 0;
    unsigned khz = 0;
    if (fscanf(f, "%u", &khz) != 1) khz = 0;
    fclose(f);
    return khz;
}

// (1) pure dpbusd throughput: 8 chains, 4 register-resident operand pairs
static uint64_t t_pure(int rep) {
    __m512i s0[4], s1[4];
    for (int i = 0; i < 4; i++) { s0[i] = _mm512_set1_epi8((char) (0x11 * (i + 1))); s1[i] = _mm512_set1_epi8((char) (0x03 * (i + 1))); }
    __m512i acc[8];
    for (int t = 0; t < 8; t++) acc[t] = _mm512_setzero_si512();
    uint64_t t0 = rdtsc();
    for (int i = 0; i < rep; i++) {
        #pragma GCC ivdep
        for (int t = 0; t < 8; t++) acc[t] = _mm512_dpbusd_epi32(acc[t], s0[t & 3], s1[t & 3]);
    }
    uint64_t t1 = rdtsc();
    __m512i s = acc[0];
    for (int t = 1; t < 8; t++) s = _mm512_add_epi32(s, acc[t]);
    sink_write(s);
    return t1 - t0;
}

// (2) dpbusd latency: single dependent chain
static uint64_t t_latency(int rep) {
    const __m512i s0 = _mm512_set1_epi8(0x11);
    const __m512i s1 = _mm512_set1_epi8(0x03);
    __m512i acc = _mm512_setzero_si512();
    uint64_t t0 = rdtsc();
    for (int i = 0; i < rep; i++) {
        acc = _mm512_dpbusd_epi32(acc, s0, s1);
    }
    uint64_t t1 = rdtsc();
    sink_write(acc);
    return t1 - t0;
}

// (3) per-dpbusd 4B load + vpbroadcastd from GPR (kernel's current u4b build)
static uint64_t t_bcast_gpr(const int * q0, int rep) {
    __m512i s1[4];
    for (int i = 0; i < 4; i++) s1[i] = _mm512_set1_epi8((char) (0x03 * (i + 1)));
    __m512i acc[8];
    for (int t = 0; t < 8; t++) acc[t] = _mm512_setzero_si512();
    uint64_t t0 = rdtsc();
    for (int i = 0; i < rep; i++) {
        #pragma GCC ivdep
        for (int t = 0; t < 8; t++) {
            const int u4 = q0[(i + t) & 255];
            const __m512i u4b = _mm512_set1_epi32(u4);
            acc[t] = _mm512_dpbusd_epi32(acc[t], u4b, s1[t & 3]);
        }
    }
    uint64_t t1 = rdtsc();
    __m512i s = acc[0];
    for (int t = 1; t < 8; t++) s = _mm512_add_epi32(s, acc[t]);
    sink_write(s);
    return t1 - t0;
}

// (4) per-dpbusd memory-source vpbroadcastd (the proposed one-instruction u4b)
// (GCC has no 512-bit memory broadcast intrinsic for epi32; inline asm)
static inline __m512i bcast_load_epi32(const void * p) {
    __m512i r;
    asm volatile("vpbroadcastd %1, %0" : "=v"(r) : "m"(*((const int *) p)));
    return r;
}

static uint64_t t_bcast_mem(const int * q0, int rep) {
    __m512i s1[4];
    for (int i = 0; i < 4; i++) s1[i] = _mm512_set1_epi8((char) (0x03 * (i + 1)));
    __m512i acc[8];
    for (int t = 0; t < 8; t++) acc[t] = _mm512_setzero_si512();
    uint64_t t0 = rdtsc();
    for (int i = 0; i < rep; i++) {
        #pragma GCC ivdep
        for (int t = 0; t < 8; t++) {
            const __m512i u4b = bcast_load_epi32(&q0[(i + t) & 255]);
            acc[t] = _mm512_dpbusd_epi32(acc[t], u4b, s1[t & 3]);
        }
    }
    uint64_t t1 = rdtsc();
    __m512i s = acc[0];
    for (int t = 1; t < 8; t++) s = _mm512_add_epi32(s, acc[t]);
    sink_write(s);
    return t1 - t0;
}

// (5)+(6) full kernel subblock shape, NA=4: per subblock = acc16 zero, then
// 8 x {64B qv load + 4 x {4B load + set1 + dpbusd into acc16}}, then the int
// split-pass correction (mullo/padd into S1/S2, 2 set1 per t).
// WITH_CORR adds the correction. 32 dpbusd per rep.
template <bool WITH_CORR>
static uint64_t t_subblock(const unsigned char * qvb, const int * q0, const int * scp, const int * mnp, int rep) {
    __m512i S1[4], S2[4];
    __m512i acc[4];
    for (int t = 0; t < 4; t++) { S1[t] = _mm512_setzero_si512(); if (WITH_CORR) S2[t] = _mm512_setzero_si512(); acc[t] = _mm512_setzero_si512(); }
    uint64_t t0 = rdtsc();
    for (int i = 0; i < rep; i++) {
        for (int t = 0; t < 4; t++) acc[t] = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            const __m512i qv = _mm512_loadu_si512((const __m512i *) (qvb + g * 64));
            for (int t = 0; t < 4; t++) {
                const int u4 = q0[((i * 8 + g) * 4 + t) & 255];
                const __m512i u4b = _mm512_set1_epi32(u4);
                acc[t] = _mm512_dpbusd_epi32(acc[t], u4b, qv);
            }
        }
        if (WITH_CORR) {
            const __m512i bs32 = _mm512_set1_epi32(1234);
            for (int t = 0; t < 4; t++) {
                S1[t] = _mm512_add_epi32(S1[t], _mm512_mullo_epi32(acc[t], _mm512_set1_epi32(scp[t & 3])));
                S2[t] = _mm512_add_epi32(S2[t], _mm512_mullo_epi32(bs32, _mm512_set1_epi32(mnp[t & 3])));
            }
        } else {
            // checksum: keep every iteration's dpbusd output live (S1 spans the
            // rep loop), otherwise the per-iteration acc reset makes all but the
            // last iteration dead and the loop shrinks to one trip
            for (int t = 0; t < 4; t++) S1[t] = _mm512_add_epi32(S1[t], acc[t]);
        }
    }
    uint64_t t1 = rdtsc();
    // sink everything: S1/S2 (alive only with correction) and the last acc set,
    // so the dpbusd loop cannot be deleted as dead code
    __m512i s = S1[0];
    for (int t = 1; t < 4; t++) s = _mm512_add_epi32(s, S1[t]);
    for (int t = 0; t < 4; t++) {
        s = _mm512_add_epi32(s, acc[t]);
        if (WITH_CORR) s = _mm512_add_epi32(s, S2[t]);
    }
    sink_write(s);
    return t1 - t0;
}

struct result {
    const char * name;
    double cyc_per_dpbusd;
    double flops_cyc;
    double tf_core;
    unsigned freq_mhz;
};

static void report(const char * name, uint64_t cycles, int ndpbusd, const char * cpu, result * out) {
    const unsigned f = out->freq_mhz;
    const double ghz = f / 1e3; // f in MHz
    const double cpo = (double) cycles / ndpbusd;
    out->name = name;
    out->cyc_per_dpbusd = cpo;
    out->flops_cyc = 128.0 / cpo;
    out->tf_core = 128.0 * ghz / cpo / 1e3;
    printf("%-34s %8.2f cyc/dpbusd  %6.1f FLOP/c  %6.2f TF/core @ %u MHz\n",
           name, cpo, 128.0 / cpo, out->tf_core, f);
}

int main(int argc, char ** argv) {
    const char * cpu = "8";
    int rep = 1 << 22;
    if (argc > 1) cpu = argv[1];
    if (argc > 2) rep = atoi(argv[2]);

    unsigned f0 = freq_khz(cpu);

    // data: L1-resident, patterned
    static unsigned char qvb[8 * 64];
    static int q0[256], scp[4], mnp[4];
    for (int i = 0; i < 8 * 64; i++) qvb[i] = (unsigned char) (i * 7 + 3);
    for (int i = 0; i < 256; i++) q0[i] = (i * 2654435761u) & 0xFFFFFFFFu;
    scp[0] = 3; scp[1] = -7; scp[2] = 12; scp[3] = -1;
    mnp[0] = 2; mnp[1] = -3; mnp[2] = 5; mnp[3] = 0;

    warmup();

    result r[6];
    for (int i = 0; i < 6; i++) r[i].freq_mhz = freq_khz(cpu) / 1000;

    printf("cpu%s: warm freq %u MHz -> post-warmup %u MHz\n\n", cpu, f0 / 1000, r[0].freq_mhz);

    uint64_t c;

    c = t_pure(rep);
    r[0].freq_mhz = freq_khz(cpu) / 1000;
    report("pure dpbusd (8 chains)", c, rep * 8, cpu, &r[0]);

    c = t_latency(1 << 23);
    r[1].freq_mhz = freq_khz(cpu) / 1000;
    report("dpbusd latency (1 chain)", c, 1 << 23, cpu, &r[1]);

    c = t_bcast_gpr(q0, rep);
    r[2].freq_mhz = freq_khz(cpu) / 1000;
    report("dpbusd + 4B load + bcast GPR", c, rep * 8, cpu, &r[2]);

    c = t_bcast_mem(q0, rep);
    r[3].freq_mhz = freq_khz(cpu) / 1000;
    report("dpbusd + bcast load m32", c, rep * 8, cpu, &r[3]);

    c = t_subblock<false>(qvb, q0, scp, mnp, rep);
    r[4].freq_mhz = freq_khz(cpu) / 1000;
    report("kernel subblock shape (NA=4)", c, rep * 32, cpu, &r[4]);

    c = t_subblock<true>(qvb, q0, scp, mnp, rep);
    r[5].freq_mhz = freq_khz(cpu) / 1000;
    report("kernel subblock + correction", c, rep * 32, cpu, &r[5]);

    // the numbers that matter
    printf("\npeak ceiling (pure):      %.2f TF/core  (%.1f FLOP/c)\n", r[0].tf_core, r[0].flops_cyc);
    printf("achieved (kernel shape+cor): %.2f TF/core  -> %d%% of peak\n",
           r[5].tf_core, (int) (100.0 * r[5].tf_core / r[0].tf_core));
    printf("bcast cost (3 vs 1):        %.2f cyc/op added per dpbusd\n", r[2].cyc_per_dpbusd - r[0].cyc_per_dpbusd);
    printf("mem-bcast vs GPR-bcast:     %.2f cyc/op\n", r[3].cyc_per_dpbusd - r[2].cyc_per_dpbusd);
    printf("correction cost (6 vs 5):   %.2f cyc/dpbusd\n", r[5].cyc_per_dpbusd - r[4].cyc_per_dpbusd);
    return 0;
}