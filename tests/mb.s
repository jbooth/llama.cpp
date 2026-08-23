	.file	"microbench-dpbusd.cpp"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC0:
	.string	"/sys/devices/system/cpu/cpu%s/cpufreq/scaling_cur_freq"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	"r"
.LC2:
	.string	"%u"
	.text
	.p2align 4
	.type	_ZL8freq_khzPKc, @function
_ZL8freq_khzPKc:
.LFB6645:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %r9
	movl	$256, %esi
	leaq	.LC0(%rip), %r8
	movl	$256, %ecx
	movl	$2, %edx
	subq	$288, %rsp
	.cfi_def_cfa_offset 304
	movq	%fs:40, %rax
	movq	%rax, 280(%rsp)
	xorl	%eax, %eax
	leaq	16(%rsp), %rbx
	movq	%rbx, %rdi
	call	__snprintf_chk@PLT
	leaq	.LC1(%rip), %rsi
	movq	%rbx, %rdi
	call	fopen@PLT
	testq	%rax, %rax
	je	.L5
	movq	%rax, %rbx
	movq	%rax, %rdi
	leaq	12(%rsp), %rdx
	xorl	%eax, %eax
	leaq	.LC2(%rip), %rsi
	movl	$0, 12(%rsp)
	call	__isoc23_fscanf@PLT
	cmpl	$1, %eax
	je	.L3
	movl	$0, 12(%rsp)
.L3:
	movq	%rbx, %rdi
	call	fclose@PLT
	movl	12(%rsp), %eax
.L1:
	movq	280(%rsp), %rdx
	subq	%fs:40, %rdx
	jne	.L11
	addq	$288, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
.L5:
	.cfi_restore_state
	xorl	%eax, %eax
	jmp	.L1
.L11:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE6645:
	.size	_ZL8freq_khzPKc, .-_ZL8freq_khzPKc
	.section	.rodata.str1.8
	.align 8
.LC6:
	.string	"%-34s %8.2f cyc/dpbusd  %6.1f FLOP/c  %6.2f TF/core @ %u MHz\n"
	.text
	.p2align 4
	.type	_ZL6reportPKcmiS0_P6result.isra.0, @function
_ZL6reportPKcmiS0_P6result.isra.0:
.LFB6657:
	.cfi_startproc
	vxorps	%xmm3, %xmm3, %xmm3
	movl	32(%rcx), %eax
	movq	%rdi, (%rcx)
	vmovsd	.LC3(%rip), %xmm2
	vcvtsi2sdl	%edx, %xmm3, %xmm1
	movq	%rdi, %rdx
	movl	$2, %edi
	vcvtusi2sdq	%rsi, %xmm3, %xmm0
	leaq	.LC6(%rip), %rsi
	vcvtusi2sdl	%eax, %xmm3, %xmm3
	vdivsd	.LC4(%rip), %xmm3, %xmm3
	vdivsd	%xmm1, %xmm0, %xmm0
	vdivsd	%xmm0, %xmm2, %xmm1
	vmulsd	%xmm2, %xmm3, %xmm2
	vdivsd	%xmm0, %xmm2, %xmm2
	vunpcklpd	%xmm1, %xmm0, %xmm4
	vmovupd	%xmm4, 8(%rcx)
	vdivsd	.LC5(%rip), %xmm2, %xmm2
	vmovsd	%xmm2, 24(%rcx)
	movl	%eax, %ecx
	movl	$3, %eax
	jmp	__printf_chk@PLT
	.cfi_endproc
.LFE6657:
	.size	_ZL6reportPKcmiS0_P6result.isra.0, .-_ZL6reportPKcmiS0_P6result.isra.0
	.section	.rodata.str1.1
.LC7:
	.string	"8"
	.section	.rodata.str1.8
	.align 8
.LC34:
	.string	"cpu%s: warm freq %u MHz -> post-warmup %u MHz\n\n"
	.section	.rodata.str1.1
.LC43:
	.string	"pure dpbusd (8 chains)"
.LC44:
	.string	"dpbusd latency (1 chain)"
.LC45:
	.string	"dpbusd + 4B load + bcast GPR"
.LC46:
	.string	"dpbusd + bcast load m32"
.LC47:
	.string	"kernel subblock shape (NA=4)"
.LC49:
	.string	"kernel subblock + correction"
	.section	.rodata.str1.8
	.align 8
.LC50:
	.string	"\npeak ceiling (pure):      %.2f TF/core  (%.1f FLOP/c)\n"
	.align 8
.LC52:
	.string	"achieved (kernel shape+cor): %.2f TF/core  -> %d%% of peak\n"
	.align 8
.LC53:
	.string	"bcast cost (3 vs 1):        %.2f cyc/op added per dpbusd\n"
	.align 8
.LC54:
	.string	"mem-bcast vs GPR-bcast:     %.2f cyc/op\n"
	.align 8
.LC55:
	.string	"correction cost (6 vs 5):   %.2f cyc/dpbusd\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB6653:
	.cfi_startproc
	endbr64
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-64, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	subq	$1984, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	leaq	.LC7(%rip), %rax
	movl	$4194304, -1776(%rbp)
	movq	%rax, -1656(%rbp)
	cmpl	$1, %edi
	jle	.L14
	movq	8(%rsi), %rax
	movq	%rax, -1656(%rbp)
	cmpl	$2, %edi
	jne	.L47
.L14:
	movq	-1656(%rbp), %rdi
	call	_ZL8freq_khzPKc
	vmovdqa	.LC15(%rip), %ymm0
	vmovdqa	.LC8(%rip), %ymm7
	vmovdqa	.LC9(%rip), %ymm6
	movl	%eax, -1840(%rbp)
	xorl	%eax, %eax
	vmovdqa	%ymm0, 224+_ZZ4mainE2bz(%rip)
	vmovdqa	.LC10(%rip), %ymm5
	vmovdqa	.LC11(%rip), %ymm4
	vmovdqa	%ymm0, 480+_ZZ4mainE2bz(%rip)
	vmovdqa32	.LC16(%rip), %zmm0
	vmovdqa	.LC12(%rip), %ymm3
	vmovdqa	.LC13(%rip), %ymm2
	vmovdqa	.LC14(%rip), %ymm1
	vmovdqa	%ymm7, _ZZ4mainE2bz(%rip)
	vmovdqa32	%zmm0, _ZZ4mainE2ap(%rip)
	vmovdqa32	.LC17(%rip), %zmm0
	vmovdqa	%ymm6, 32+_ZZ4mainE2bz(%rip)
	vmovdqa32	%zmm0, 64+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC18(%rip), %zmm0
	vmovdqa	%ymm5, 64+_ZZ4mainE2bz(%rip)
	vmovdqa32	%zmm0, 128+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC19(%rip), %zmm0
	vmovdqa	%ymm4, 96+_ZZ4mainE2bz(%rip)
	vmovdqa	%ymm3, 128+_ZZ4mainE2bz(%rip)
	vmovdqa	%ymm2, 160+_ZZ4mainE2bz(%rip)
	vmovdqa	%ymm1, 192+_ZZ4mainE2bz(%rip)
	vmovdqa	%ymm7, 256+_ZZ4mainE2bz(%rip)
	vmovdqa	%ymm6, 288+_ZZ4mainE2bz(%rip)
	vmovdqa	%ymm5, 320+_ZZ4mainE2bz(%rip)
	vmovdqa	%ymm4, 352+_ZZ4mainE2bz(%rip)
	vmovdqa	%ymm3, 384+_ZZ4mainE2bz(%rip)
	vmovdqa	%ymm2, 416+_ZZ4mainE2bz(%rip)
	vmovdqa	%ymm1, 448+_ZZ4mainE2bz(%rip)
	vmovdqa32	%zmm0, 192+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC20(%rip), %zmm0
	movq	$0, -1640(%rbp)
	vmovdqa32	%zmm0, 256+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC21(%rip), %zmm0
	vmovdqa32	%zmm0, 320+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC22(%rip), %zmm0
	vmovdqa32	%zmm0, 384+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC23(%rip), %zmm0
	vmovdqa32	%zmm0, 448+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC24(%rip), %zmm0
	vmovdqa32	%zmm0, 512+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC25(%rip), %zmm0
	vmovdqa32	%zmm0, 576+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC26(%rip), %zmm0
	vmovdqa32	%zmm0, 640+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC27(%rip), %zmm0
	vmovdqa32	%zmm0, 704+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC28(%rip), %zmm0
	vmovdqa32	%zmm0, 768+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC29(%rip), %zmm0
	vmovdqa32	%zmm0, 832+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC30(%rip), %zmm0
	vmovdqa32	%zmm0, 896+_ZZ4mainE2ap(%rip)
	vmovdqa32	.LC31(%rip), %zmm0
	vmovdqa32	%zmm0, 960+_ZZ4mainE2ap(%rip)
	vmovdqa	.LC32(%rip), %xmm0
	vmovdqa	%xmm0, _ZZ4mainE3scp(%rip)
	vmovdqa	.LC33(%rip), %xmm0
	vmovdqa	%xmm0, _ZZ4mainE3mnp(%rip)
	.p2align 5
	.p2align 4
	.p2align 3
.L15:
	movq	-1640(%rbp), %rdx
	addq	%rax, %rdx
	addq	$1, %rax
	movq	%rdx, -1640(%rbp)
	cmpq	$16777216, %rax
	jne	.L15
	movq	-1640(%rbp), %rax
	leaq	-1632(%rbp), %rax
	leaq	-1392(%rbp), %r13
	movq	%rax, -1904(%rbp)
	movq	%rax, %rbx
	leaq	-320(%rbp), %r12
	leaq	.LC1(%rip), %r14
	vzeroupper
	.p2align 4
	.p2align 3
.L18:
	movl	$256, %ecx
	movl	$2, %edx
	movq	%r12, %rdi
	xorl	%eax, %eax
	movq	-1656(%rbp), %r9
	leaq	.LC0(%rip), %r8
	movl	$256, %esi
	call	__snprintf_chk@PLT
	movq	%r14, %rsi
	movq	%r12, %rdi
	call	fopen@PLT
	movq	%rax, %r15
	testq	%rax, %rax
	je	.L34
	movq	%rax, %rdi
	xorl	%eax, %eax
	leaq	-1644(%rbp), %rdx
	movl	$0, -1644(%rbp)
	leaq	.LC2(%rip), %rsi
	call	__isoc23_fscanf@PLT
	cmpl	$1, %eax
	je	.L17
	movl	$0, -1644(%rbp)
.L17:
	movq	%r15, %rdi
	call	fclose@PLT
	movl	-1644(%rbp), %eax
.L16:
	imulq	$274877907, %rax, %rax
	addq	$40, %rbx
	shrq	$38, %rax
	movl	%eax, -8(%rbx)
	cmpq	%r13, %rbx
	jne	.L18
	movl	-1840(%rbp), %ecx
	movl	-1600(%rbp), %r8d
	xorl	%eax, %eax
	leaq	.LC34(%rip), %rsi
	movq	-1656(%rbp), %rdx
	movl	$2, %edi
	leaq	-880(%rbp), %rbx
	imulq	$274877907, %rcx, %rcx
	shrq	$38, %rcx
	call	__printf_chk@PLT
	xorl	%eax, %eax
	movq	%rbx, %rdi
	movl	$64, %ecx
	rep stosq
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	salq	$32, %rdx
	movl	-1776(%rbp), %esi
	movl	%eax, %eax
	movq	%rdx, %rcx
	orq	%rax, %rcx
	testl	%esi, %esi
	jle	.L19
	movl	$286331153, %edx
	xorl	%eax, %eax
	vmovdqa64	-880(%rbp), %zmm23
	vmovdqa64	-816(%rbp), %zmm22
	vmovdqa64	-752(%rbp), %zmm21
	vpbroadcastd	%edx, %zmm7
	movl	$50529027, %edx
	vmovdqa64	-688(%rbp), %zmm20
	vmovdqa64	-624(%rbp), %zmm19
	vpbroadcastd	%edx, %zmm6
	movl	$572662306, %edx
	vmovdqa64	-560(%rbp), %zmm18
	vmovdqa64	-496(%rbp), %zmm17
	vpbroadcastd	%edx, %zmm5
	movl	$101058054, %edx
	vmovdqa64	-432(%rbp), %zmm16
	vpbroadcastd	%edx, %zmm4
	movl	$858993459, %edx
	vpbroadcastd	%edx, %zmm3
	movl	$151587081, %edx
	vpbroadcastd	%edx, %zmm2
	movl	$1145324612, %edx
	vpbroadcastd	%edx, %zmm1
	movl	$202116108, %edx
	vpbroadcastd	%edx, %zmm0
	movl	%esi, %edx
	.p2align 4
	.p2align 3
.L20:
	vmovdqa32	%zmm23, %zmm15
	vmovdqa32	%zmm22, %zmm14
	addl	$1, %eax
	vmovdqa32	%zmm21, %zmm13
	vmovdqa32	%zmm20, %zmm12
	vmovdqa32	%zmm19, %zmm11
	vpdpbusd	%zmm6, %zmm7, %zmm15
	vmovdqa32	%zmm18, %zmm10
	vmovdqa32	%zmm17, %zmm9
	vmovdqa32	%zmm16, %zmm8
	vpdpbusd	%zmm4, %zmm5, %zmm14
	vpdpbusd	%zmm2, %zmm3, %zmm13
	vpdpbusd	%zmm0, %zmm1, %zmm12
	vpdpbusd	%zmm6, %zmm7, %zmm11
	vpdpbusd	%zmm4, %zmm5, %zmm10
	vpdpbusd	%zmm2, %zmm3, %zmm9
	vmovdqa64	%zmm15, %zmm23
	vpdpbusd	%zmm0, %zmm1, %zmm8
	vmovdqa64	%zmm14, %zmm22
	vmovdqa64	%zmm13, %zmm21
	vmovdqa64	%zmm12, %zmm20
	vmovdqa64	%zmm11, %zmm19
	vmovdqa64	%zmm10, %zmm18
	vmovdqa64	%zmm9, %zmm17
	vmovdqa64	%zmm8, %zmm16
	cmpl	%eax, %edx
	jne	.L20
	vmovdqa64	%zmm15, -880(%rbp)
	vmovdqa64	%zmm14, -816(%rbp)
	vmovdqa64	%zmm13, -752(%rbp)
	vmovdqa64	%zmm12, -688(%rbp)
	vmovdqa64	%zmm11, -624(%rbp)
	vmovdqa64	%zmm10, -560(%rbp)
	vmovdqa64	%zmm9, -496(%rbp)
	vmovdqa64	%zmm8, -432(%rbp)
.L19:
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	vmovdqa32	-816(%rbp), %zmm0
	salq	$32, %rdx
	movl	%eax, %eax
	movq	-1656(%rbp), %rdi
	vpaddd	-880(%rbp), %zmm0, %zmm0
	orq	%rax, %rdx
	vpaddd	-624(%rbp), %zmm0, %zmm0
	vmovdqa32	-688(%rbp), %zmm1
	subq	%rcx, %rdx
	vpaddd	-496(%rbp), %zmm0, %zmm0
	vpaddd	-752(%rbp), %zmm1, %zmm1
	movq	%rdx, %r13
	vpaddd	-560(%rbp), %zmm1, %zmm1
	vpaddd	-432(%rbp), %zmm1, %zmm1
	vpaddd	%zmm1, %zmm0, %zmm0
	vextracti64x4	$0x1, %zmm0, %ymm1
	vpaddd	%ymm0, %ymm1, %ymm0
	vextracti64x2	$0x1, %ymm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm0
	vpshufd	$78, %xmm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm1
	vmovdqa	%xmm1, %xmm0
	vpshufd	$85, %xmm1, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vmovd	%xmm0, %esi
	movslq	%esi, %rsi
	movq	%rsi, _ZL6sink_v(%rip)
	vzeroupper
	call	_ZL8freq_khzPKc
	movq	-1904(%rbp), %rcx
	movq	%r13, %rsi
	movl	%eax, %eax
	leaq	.LC43(%rip), %rdi
	imulq	$274877907, %rax, %rax
	shrq	$38, %rax
	movl	%eax, -1600(%rbp)
	movl	-1776(%rbp), %eax
	leal	0(,%rax,8), %r12d
	movl	%r12d, %edx
	call	_ZL6reportPKcmiS0_P6result.isra.0
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	salq	$32, %rdx
	movl	%eax, %eax
	movl	$8388608, %r13d
	movq	%rdx, %rsi
	vpxor	%xmm1, %xmm1, %xmm1
	orq	%rax, %rsi
	movl	$286331153, %eax
	vpbroadcastd	%eax, %zmm3
	movl	$50529027, %eax
	vpbroadcastd	%eax, %zmm2
	.p2align 5
	.p2align 4
	.p2align 3
.L21:
	vmovdqa32	%zmm1, %zmm0
	vpdpbusd	%zmm2, %zmm3, %zmm0
	vmovdqa64	%zmm0, %zmm1
	subl	$1, %r13d
	jne	.L21
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	vextracti64x4	$0x1, %zmm0, %ymm1
	salq	$32, %rdx
	movl	%eax, %eax
	movq	-1656(%rbp), %rdi
	vpaddd	%ymm0, %ymm1, %ymm0
	orq	%rax, %rdx
	vextracti64x2	$0x1, %ymm0, %xmm1
	subq	%rsi, %rdx
	vpaddd	%xmm0, %xmm1, %xmm0
	movq	%rdx, %r14
	vpshufd	$78, %xmm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm1
	vmovdqa	%xmm1, %xmm0
	vpshufd	$85, %xmm1, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vmovd	%xmm0, %ecx
	movslq	%ecx, %rcx
	movq	%rcx, _ZL6sink_v(%rip)
	vzeroupper
	call	_ZL8freq_khzPKc
	movl	$8388608, %edx
	leaq	-1592(%rbp), %rcx
	movl	%eax, %eax
	movq	%r14, %rsi
	leaq	.LC44(%rip), %rdi
	imulq	$274877907, %rax, %rax
	shrq	$38, %rax
	movl	%eax, -1560(%rbp)
	call	_ZL6reportPKcmiS0_P6result.isra.0
	xorl	%eax, %eax
	movq	%rbx, %rdi
	movl	$64, %ecx
	rep stosq
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	salq	$32, %rdx
	movl	-1776(%rbp), %r8d
	movl	%eax, %eax
	movq	%rdx, %rsi
	orq	%rax, %rsi
	testl	%r8d, %r8d
	jle	.L22
	movl	$50529027, %edx
	vmovdqa64	-880(%rbp), %zmm20
	vmovdqa64	-816(%rbp), %zmm19
	leaq	_ZZ4mainE2ap(%rip), %rax
	vmovdqa64	-752(%rbp), %zmm18
	vpbroadcastd	%edx, %zmm4
	movl	$101058054, %edx
	vmovdqa64	-688(%rbp), %zmm17
	vmovdqa64	-624(%rbp), %zmm16
	vpbroadcastd	%edx, %zmm3
	movl	$151587081, %edx
	vmovdqa64	-560(%rbp), %zmm15
	vmovdqa64	-496(%rbp), %zmm14
	vpbroadcastd	%edx, %zmm2
	vmovdqa64	-432(%rbp), %zmm13
	movl	$202116108, %edx
	vpbroadcastd	_ZZ4mainE2ap(%rip), %zmm0
	vpbroadcastd	%edx, %zmm1
	.p2align 4
	.p2align 3
.L23:
	movl	%ecx, %edx
	addl	$1, %ecx
	vmovdqa32	%zmm20, %zmm12
	movzbl	%cl, %edi
	vmovdqa32	%zmm18, %zmm10
	vmovdqa32	%zmm17, %zmm9
	vpdpbusd	%zmm4, %zmm0, %zmm12
	vpbroadcastd	(%rax,%rdi,4), %zmm0
	leal	2(%rdx), %edi
	vmovdqa32	%zmm16, %zmm8
	movzbl	%dil, %edi
	vmovdqa32	%zmm15, %zmm7
	vmovdqa32	%zmm14, %zmm6
	vpbroadcastd	(%rax,%rdi,4), %zmm5
	leal	3(%rdx), %edi
	vmovdqa32	%zmm19, %zmm11
	vpdpbusd	%zmm3, %zmm0, %zmm11
	movzbl	%dil, %edi
	vpdpbusd	%zmm2, %zmm5, %zmm10
	vpbroadcastd	(%rax,%rdi,4), %zmm5
	leal	4(%rdx), %edi
	movzbl	%dil, %edi
	vpdpbusd	%zmm1, %zmm5, %zmm9
	vpbroadcastd	(%rax,%rdi,4), %zmm5
	leal	5(%rdx), %edi
	vmovdqa64	%zmm12, %zmm20
	movzbl	%dil, %edi
	vpdpbusd	%zmm4, %zmm5, %zmm8
	vpbroadcastd	(%rax,%rdi,4), %zmm5
	leal	6(%rdx), %edi
	addl	$7, %edx
	movzbl	%dil, %edi
	movzbl	%dl, %edx
	vmovdqa64	%zmm11, %zmm19
	vpdpbusd	%zmm3, %zmm5, %zmm7
	vpbroadcastd	(%rax,%rdi,4), %zmm5
	vpbroadcastd	(%rax,%rdx,4), %zmm21
	vmovdqa64	%zmm10, %zmm18
	vpdpbusd	%zmm2, %zmm5, %zmm6
	vmovdqa32	%zmm13, %zmm5
	vmovdqa64	%zmm9, %zmm17
	vpdpbusd	%zmm1, %zmm21, %zmm5
	vmovdqa64	%zmm8, %zmm16
	vmovdqa64	%zmm7, %zmm15
	vmovdqa64	%zmm6, %zmm14
	vmovdqa64	%zmm5, %zmm13
	cmpl	%ecx, %r8d
	jne	.L23
	vmovdqa64	%zmm12, -880(%rbp)
	vmovdqa64	%zmm11, -816(%rbp)
	vmovdqa64	%zmm10, -752(%rbp)
	vmovdqa64	%zmm9, -688(%rbp)
	vmovdqa64	%zmm8, -624(%rbp)
	vmovdqa64	%zmm7, -560(%rbp)
	vmovdqa64	%zmm6, -496(%rbp)
	vmovdqa64	%zmm5, -432(%rbp)
.L22:
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	vmovdqa32	-880(%rbp), %zmm0
	salq	$32, %rdx
	movl	%eax, %eax
	movq	-1656(%rbp), %rdi
	vpaddd	-816(%rbp), %zmm0, %zmm0
	orq	%rax, %rdx
	vpaddd	-624(%rbp), %zmm0, %zmm0
	vmovdqa32	-752(%rbp), %zmm1
	subq	%rsi, %rdx
	vpaddd	-496(%rbp), %zmm0, %zmm0
	vpaddd	-688(%rbp), %zmm1, %zmm1
	movq	%rdx, %r14
	vpaddd	-560(%rbp), %zmm1, %zmm1
	vpaddd	-432(%rbp), %zmm1, %zmm1
	vpaddd	%zmm1, %zmm0, %zmm0
	vextracti64x4	$0x1, %zmm0, %ymm1
	vpaddd	%ymm0, %ymm1, %ymm0
	vextracti64x2	$0x1, %ymm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm0
	vpshufd	$78, %xmm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm1
	vmovdqa	%xmm1, %xmm0
	vpshufd	$85, %xmm1, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vmovd	%xmm0, %ecx
	movslq	%ecx, %rcx
	movq	%rcx, _ZL6sink_v(%rip)
	vzeroupper
	call	_ZL8freq_khzPKc
	movl	%r12d, %edx
	leaq	-1552(%rbp), %rcx
	movl	%eax, %eax
	movq	%r14, %rsi
	leaq	.LC45(%rip), %rdi
	imulq	$274877907, %rax, %rax
	shrq	$38, %rax
	movl	%eax, -1520(%rbp)
	call	_ZL6reportPKcmiS0_P6result.isra.0
	xorl	%eax, %eax
	movq	%rbx, %rdi
	movl	$64, %ecx
	rep stosq
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	salq	$32, %rdx
	movl	%eax, %eax
	orq	%rax, %rdx
	movq	%rdx, %r8
	movl	-1776(%rbp), %edx
	testl	%edx, %edx
	jle	.L28
	movl	$50529027, %edx
	movl	-1776(%rbp), %r9d
	xorl	%ecx, %ecx
	xorl	%esi, %esi
	vpbroadcastd	%edx, %zmm4
	movl	$101058054, %edx
	vmovdqa64	-880(%rbp), %zmm12
	vmovdqa64	-816(%rbp), %zmm11
	vpbroadcastd	%edx, %zmm3
	movl	$151587081, %edx
	vmovdqa64	-752(%rbp), %zmm10
	vmovdqa64	-688(%rbp), %zmm9
	vpbroadcastd	%edx, %zmm2
	movl	$202116108, %edx
	vmovdqa64	-624(%rbp), %zmm8
	vmovdqa64	-560(%rbp), %zmm7
	vmovdqa64	-496(%rbp), %zmm6
	leaq	_ZZ4mainE2ap(%rip), %rax
	vmovdqa64	-432(%rbp), %zmm5
	vpbroadcastd	%edx, %zmm1
	.p2align 4
	.p2align 3
.L27:
#APP
# 118 "microbench-dpbusd.cpp" 1
	vpbroadcastd (%rax,%rsi,4), %zmm13
# 0 "" 2
#NO_APP
	vmovdqa32	%zmm12, %zmm0
	movl	%ecx, %edx
	addl	$1, %ecx
	vpdpbusd	%zmm4, %zmm13, %zmm0
	movzbl	%cl, %esi
	movzbl	%cl, %edi
	vmovdqa64	%zmm0, -880(%rbp)
	vmovdqa64	%zmm0, %zmm12
#APP
# 118 "microbench-dpbusd.cpp" 1
	vpbroadcastd (%rax,%rdi,4), %zmm13
# 0 "" 2
#NO_APP
	vmovdqa32	%zmm11, %zmm0
	leal	2(%rdx), %edi
	vpdpbusd	%zmm3, %zmm13, %zmm0
	movzbl	%dil, %edi
	vmovdqa64	%zmm0, -816(%rbp)
	vmovdqa64	%zmm0, %zmm11
#APP
# 118 "microbench-dpbusd.cpp" 1
	vpbroadcastd (%rax,%rdi,4), %zmm13
# 0 "" 2
#NO_APP
	vmovdqa32	%zmm10, %zmm0
	leal	3(%rdx), %edi
	vpdpbusd	%zmm2, %zmm13, %zmm0
	movzbl	%dil, %edi
	vmovdqa64	%zmm0, -752(%rbp)
	vmovdqa64	%zmm0, %zmm10
#APP
# 118 "microbench-dpbusd.cpp" 1
	vpbroadcastd (%rax,%rdi,4), %zmm13
# 0 "" 2
#NO_APP
	vmovdqa32	%zmm9, %zmm0
	leal	4(%rdx), %edi
	vpdpbusd	%zmm1, %zmm13, %zmm0
	movzbl	%dil, %edi
	vmovdqa64	%zmm0, -688(%rbp)
	vmovdqa64	%zmm0, %zmm9
#APP
# 118 "microbench-dpbusd.cpp" 1
	vpbroadcastd (%rax,%rdi,4), %zmm13
# 0 "" 2
#NO_APP
	vmovdqa32	%zmm8, %zmm0
	leal	5(%rdx), %edi
	vpdpbusd	%zmm4, %zmm13, %zmm0
	movzbl	%dil, %edi
	vmovdqa64	%zmm0, -624(%rbp)
	vmovdqa64	%zmm0, %zmm8
#APP
# 118 "microbench-dpbusd.cpp" 1
	vpbroadcastd (%rax,%rdi,4), %zmm13
# 0 "" 2
#NO_APP
	vmovdqa32	%zmm7, %zmm0
	leal	6(%rdx), %edi
	vpdpbusd	%zmm3, %zmm13, %zmm0
	movzbl	%dil, %edi
	vmovdqa64	%zmm0, -560(%rbp)
	vmovdqa64	%zmm0, %zmm7
#APP
# 118 "microbench-dpbusd.cpp" 1
	vpbroadcastd (%rax,%rdi,4), %zmm13
# 0 "" 2
#NO_APP
	vmovdqa32	%zmm6, %zmm0
	addl	$7, %edx
	vpdpbusd	%zmm2, %zmm13, %zmm0
	movzbl	%dl, %edx
	vmovdqa64	%zmm0, -496(%rbp)
	vmovdqa64	%zmm0, %zmm6
#APP
# 118 "microbench-dpbusd.cpp" 1
	vpbroadcastd (%rax,%rdx,4), %zmm13
# 0 "" 2
#NO_APP
	vmovdqa32	%zmm5, %zmm0
	vpdpbusd	%zmm1, %zmm13, %zmm0
	vmovdqa64	%zmm0, -432(%rbp)
	vmovdqa64	%zmm0, %zmm5
	cmpl	%ecx, %r9d
	jne	.L27
.L28:
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	vmovdqa32	-752(%rbp), %zmm0
	salq	$32, %rdx
	movl	%eax, %eax
	movq	-1656(%rbp), %rdi
	vpaddd	-688(%rbp), %zmm0, %zmm0
	orq	%rax, %rdx
	vpaddd	-560(%rbp), %zmm0, %zmm0
	vmovdqa32	-880(%rbp), %zmm1
	movq	%rdx, %rbx
	vpaddd	-432(%rbp), %zmm0, %zmm0
	vpaddd	-816(%rbp), %zmm1, %zmm1
	subq	%r8, %rbx
	vpaddd	-624(%rbp), %zmm1, %zmm1
	vpaddd	-496(%rbp), %zmm1, %zmm1
	vpaddd	%zmm1, %zmm0, %zmm0
	vextracti64x4	$0x1, %zmm0, %ymm1
	vpaddd	%ymm0, %ymm1, %ymm0
	vextracti64x2	$0x1, %ymm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm0
	vpshufd	$78, %xmm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm1
	vmovdqa	%xmm1, %xmm0
	vpshufd	$85, %xmm1, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vmovd	%xmm0, %ecx
	movslq	%ecx, %rcx
	movq	%rcx, _ZL6sink_v(%rip)
	vzeroupper
	call	_ZL8freq_khzPKc
	movl	%r12d, %edx
	leaq	-1512(%rbp), %rcx
	movl	%eax, %eax
	movq	%rbx, %rsi
	leaq	.LC46(%rip), %rdi
	imulq	$274877907, %rax, %rax
	shrq	$38, %rax
	movl	%eax, -1480(%rbp)
	call	_ZL6reportPKcmiS0_P6result.isra.0
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	salq	$32, %rdx
	movl	%eax, %eax
	movq	%rdx, %rcx
	orq	%rax, %rcx
	movl	-1776(%rbp), %eax
	testl	%eax, %eax
	jle	.L48
	movl	-1776(%rbp), %eax
	vpxor	%xmm1, %xmm1, %xmm1
	vmovdqu64	_ZZ4mainE2bz(%rip), %zmm9
	vmovdqu64	64+_ZZ4mainE2bz(%rip), %zmm8
	vmovdqu64	128+_ZZ4mainE2bz(%rip), %zmm7
	vmovdqu64	192+_ZZ4mainE2bz(%rip), %zmm6
	leal	-1(%rax), %edi
	vmovdqu64	256+_ZZ4mainE2bz(%rip), %zmm5
	vmovdqu64	320+_ZZ4mainE2bz(%rip), %zmm4
	movl	%edi, %edx
	vmovdqu64	384+_ZZ4mainE2bz(%rip), %zmm3
	vmovdqu64	448+_ZZ4mainE2bz(%rip), %zmm2
	sall	$5, %edx
	movl	%edx, %eax
	leal	1(%rdx), %esi
	leal	2(%rdx), %r8d
	addl	$3, %edx
	andl	$224, %eax
	movzbl	%sil, %ebx
	movzbl	%dl, %r15d
	vmovq	%rax, %xmm0
	leaq	_ZZ4mainE2ap(%rip), %rax
	vmovq	%rbx, %xmm13
	movl	%r8d, %ebx
	vmovq	%xmm0, %rsi
	leal	1(,%rdi,8), %edx
	andl	$254, %ebx
	vpbroadcastd	(%rax,%rsi,4), %zmm22
	sall	$2, %edx
	vmovdqa32	%zmm1, %zmm0
	vmovq	%rbx, %xmm10
	movl	%edx, %ebx
	leal	1(%rdx), %edi
	leal	4(%rdx), %r8d
	vpdpbusd	%zmm9, %zmm22, %zmm0
	andl	$252, %ebx
	leal	11(%rdx), %r10d
	leal	6(%rdx), %r14d
	vmovq	%rbx, %xmm21
	movzbl	%dil, %ebx
	leal	2(%rdx), %edi
	andl	$254, %r14d
	vmovq	%xmm21, %rsi
	vmovq	%rbx, %xmm12
	movl	%edi, %ebx
	vpbroadcastd	(%rax,%rsi,4), %zmm21
	andl	$254, %ebx
	leal	10(%rdx), %r12d
	leal	3(%rdx), %edi
	vmovq	%rbx, %xmm29
	movl	%r8d, %ebx
	leal	5(%rdx), %r8d
	movzbl	%dil, %edi
	andl	$252, %ebx
	andl	$254, %r12d
	vpdpbusd	%zmm8, %zmm21, %zmm0
	vmovq	%rbx, %xmm18
	movzbl	%r8b, %ebx
	leal	7(%rdx), %r8d
	vmovq	%xmm18, %rsi
	vmovq	%rbx, %xmm11
	movzbl	%r8b, %ebx
	vpbroadcastd	(%rax,%rsi,4), %zmm18
	leal	8(%rdx), %r8d
	vmovq	%rbx, %xmm25
	movl	%r8d, %ebx
	leal	9(%rdx), %r8d
	andl	$252, %ebx
	vmovq	%rbx, %xmm17
	movzbl	%r8b, %ebx
	leal	12(%rdx), %r8d
	vpdpbusd	%zmm7, %zmm18, %zmm0
	vmovq	%xmm17, %rsi
	vmovq	%rbx, %xmm20
	movzbl	%r10b, %ebx
	vpbroadcastd	(%rax,%rsi,4), %zmm17
	vmovq	%rbx, %xmm26
	movl	%r8d, %ebx
	leal	15(%rdx), %r10d
	andl	$252, %ebx
	leal	13(%rdx), %r8d
	movzbl	%r10b, %r11d
	vmovq	%rbx, %xmm16
	movzbl	%r8b, %ebx
	leal	16(%rdx), %r8d
	vmovq	%xmm16, %rsi
	vmovq	%r11, %xmm27
	movl	%r8d, %r11d
	vpbroadcastd	(%rax,%rsi,4), %zmm16
	leal	17(%rdx), %r8d
	andl	$252, %r11d
	vmovq	%rbx, %xmm19
	vpdpbusd	%zmm6, %zmm17, %zmm0
	vmovq	%r11, %xmm15
	movzbl	%r8b, %r11d
	leal	19(%rdx), %r8d
	vmovq	%xmm15, %rsi
	vmovq	%r11, %xmm23
	leal	14(%rdx), %ebx
	movzbl	%r8b, %r10d
	vpbroadcastd	(%rax,%rsi,4), %zmm15
	leal	20(%rdx), %r8d
	vmovq	%r10, %xmm28
	andl	$254, %ebx
	movl	%r8d, %r10d
	leal	21(%rdx), %r8d
	leal	18(%rdx), %r11d
	andl	$252, %r10d
	andl	$254, %r11d
	vmovq	%r10, %xmm14
	movzbl	%r8b, %r10d
	leal	23(%rdx), %r8d
	vpdpbusd	%zmm5, %zmm16, %zmm0
	vmovq	%xmm14, %rsi
	movzbl	%r8b, %r9d
	leal	24(%rdx), %r8d
	vpbroadcastd	(%rax,%rsi,4), %zmm14
	andl	$252, %r8d
	vmovq	%xmm13, %rsi
	vmovdqa32	%zmm1, %zmm13
	vmovq	%r10, %xmm24
	leal	22(%rdx), %r10d
	andl	$254, %r10d
	vpdpbusd	%zmm4, %zmm15, %zmm0
	vpdpbusd	%zmm3, %zmm14, %zmm0
	vpbroadcastd	(%rax,%r8,4), %zmm14
	leal	25(%rdx), %r8d
	movzbl	%r8b, %r8d
	vpdpbusd	%zmm2, %zmm14, %zmm0
	vpbroadcastd	(%rax,%rsi,4), %zmm14
	vmovq	%xmm12, %rsi
	vpbroadcastd	(%rax,%rsi,4), %zmm12
	vmovq	%xmm11, %rsi
	vpdpbusd	%zmm9, %zmm14, %zmm13
	vpdpbusd	%zmm8, %zmm12, %zmm13
	vpbroadcastd	(%rax,%rsi,4), %zmm12
	vmovq	%xmm20, %rsi
	vmovdqa32	%zmm13, %zmm11
	vpdpbusd	%zmm7, %zmm12, %zmm11
	vpbroadcastd	(%rax,%rsi,4), %zmm12
	vmovq	%xmm19, %rsi
	vpdpbusd	%zmm6, %zmm12, %zmm11
	vpbroadcastd	(%rax,%rsi,4), %zmm12
	vmovq	%xmm23, %rsi
	vpdpbusd	%zmm5, %zmm12, %zmm11
	vpbroadcastd	(%rax,%rsi,4), %zmm12
	vmovq	%xmm24, %rsi
	vpdpbusd	%zmm4, %zmm12, %zmm11
	vpbroadcastd	(%rax,%rsi,4), %zmm12
	vmovq	%xmm10, %rsi
	vmovdqa32	%zmm1, %zmm10
	vpdpbusd	%zmm3, %zmm12, %zmm11
	vpbroadcastd	(%rax,%r8,4), %zmm12
	leal	26(%rdx), %r8d
	addl	$27, %edx
	movzbl	%r8b, %r8d
	movzbl	%dl, %edx
	vpdpbusd	%zmm2, %zmm12, %zmm11
	vpbroadcastd	(%rax,%rsi,4), %zmm12
	vmovq	%xmm29, %rsi
	vpdpbusd	%zmm9, %zmm12, %zmm10
	vpbroadcastd	(%rax,%rsi,4), %zmm12
	vpdpbusd	%zmm8, %zmm12, %zmm10
	vpbroadcastd	(%rax,%r14,4), %zmm12
	vpdpbusd	%zmm7, %zmm12, %zmm10
	vpbroadcastd	(%rax,%r12,4), %zmm12
	vpdpbusd	%zmm6, %zmm12, %zmm10
	vpbroadcastd	(%rax,%rbx,4), %zmm12
	vmovq	%xmm25, %rbx
	vpdpbusd	%zmm5, %zmm12, %zmm10
	vpbroadcastd	(%rax,%r11,4), %zmm12
	vpdpbusd	%zmm4, %zmm12, %zmm10
	vpbroadcastd	(%rax,%r10,4), %zmm12
	vpdpbusd	%zmm3, %zmm12, %zmm10
	vpbroadcastd	(%rax,%r8,4), %zmm12
	vpdpbusd	%zmm2, %zmm12, %zmm10
	vpbroadcastd	(%rax,%r15,4), %zmm12
	vpdpbusd	%zmm9, %zmm12, %zmm1
	vpbroadcastd	(%rax,%rdi,4), %zmm9
	vpdpbusd	%zmm8, %zmm9, %zmm1
	vpbroadcastd	(%rax,%rbx,4), %zmm8
	vmovq	%xmm26, %rbx
	vpdpbusd	%zmm7, %zmm8, %zmm1
	vpbroadcastd	(%rax,%rbx,4), %zmm7
	vmovq	%xmm27, %rbx
	vpdpbusd	%zmm6, %zmm7, %zmm1
	vpbroadcastd	(%rax,%rbx,4), %zmm6
	vmovq	%xmm28, %rbx
	vpdpbusd	%zmm5, %zmm6, %zmm1
	vpbroadcastd	(%rax,%rbx,4), %zmm5
	vpdpbusd	%zmm4, %zmm5, %zmm1
	vpbroadcastd	(%rax,%r9,4), %zmm4
	vpdpbusd	%zmm3, %zmm4, %zmm1
	vpbroadcastd	(%rax,%rdx,4), %zmm3
	vpdpbusd	%zmm2, %zmm3, %zmm1
.L26:
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	vpaddd	%zmm1, %zmm10, %zmm10
	vpaddd	%zmm11, %zmm0, %zmm0
	salq	$32, %rdx
	movl	%eax, %eax
	vpaddd	%zmm10, %zmm0, %zmm0
	movq	-1656(%rbp), %rdi
	orq	%rax, %rdx
	vextracti64x4	$0x1, %zmm0, %ymm1
	subq	%rcx, %rdx
	vpaddd	%ymm0, %ymm1, %ymm0
	movq	%rdx, %r12
	vextracti64x2	$0x1, %ymm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm0
	vpshufd	$78, %xmm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm1
	vmovdqa	%xmm1, %xmm0
	vpshufd	$85, %xmm1, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vmovd	%xmm0, %esi
	movslq	%esi, %rsi
	movq	%rsi, _ZL6sink_v(%rip)
	vzeroupper
	call	_ZL8freq_khzPKc
	movl	-1776(%rbp), %r14d
	leaq	-1472(%rbp), %rcx
	movl	%eax, %eax
	movq	%r12, %rsi
	leaq	.LC47(%rip), %rdi
	imulq	$274877907, %rax, %rax
	movl	%r14d, %ebx
	sall	$5, %ebx
	movl	%ebx, %edx
	shrq	$38, %rax
	movl	%eax, -1440(%rbp)
	call	_ZL6reportPKcmiS0_P6result.isra.0
	vpxor	%xmm0, %xmm0, %xmm0
	vmovdqa64	%zmm0, -1392(%rbp)
	vmovdqa64	%zmm0, -1328(%rbp)
	vmovdqa64	%zmm0, -1264(%rbp)
	vmovdqa64	%zmm0, -1200(%rbp)
	vmovdqa64	%zmm0, -1136(%rbp)
	vmovdqa64	%zmm0, -1072(%rbp)
	vmovdqa64	%zmm0, -1008(%rbp)
	vmovdqa64	%zmm0, -944(%rbp)
	vmovdqa64	%zmm0, -880(%rbp)
	vmovdqa64	%zmm0, -816(%rbp)
	vmovdqa64	%zmm0, -752(%rbp)
	vmovdqa64	%zmm0, -688(%rbp)
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	salq	$32, %rdx
	movl	%eax, %eax
	movq	%rdx, %rcx
	orq	%rax, %rcx
	testl	%r14d, %r14d
	jle	.L29
	movl	$1234, %eax
	vpxor	%xmm8, %xmm8, %xmm8
	vpbroadcastd	8+_ZZ4mainE3scp(%rip), %zmm7
	vmovdqu32	_ZZ4mainE2bz(%rip), %zmm16
	vmovdqu32	64+_ZZ4mainE2bz(%rip), %zmm15
	vpbroadcastd	%eax, %zmm0
	vmovdqu32	128+_ZZ4mainE2bz(%rip), %zmm14
	leaq	_ZZ4mainE2ap(%rip), %rax
	vpmulld	4+_ZZ4mainE3mnp(%rip){1to16}, %zmm0, %zmm6
	vpmulld	_ZZ4mainE3mnp(%rip){1to16}, %zmm0, %zmm30
	vmovdqa32	%zmm7, -1840(%rbp)
	vpbroadcastd	12+_ZZ4mainE3scp(%rip), %zmm7
	vmovdqu32	192+_ZZ4mainE2bz(%rip), %zmm13
	vmovdqu32	256+_ZZ4mainE2bz(%rip), %zmm12
	vmovdqu32	320+_ZZ4mainE2bz(%rip), %zmm11
	vmovdqu32	384+_ZZ4mainE2bz(%rip), %zmm10
	vmovdqa32	%zmm7, -1968(%rbp)
	vmovdqu32	448+_ZZ4mainE2bz(%rip), %zmm9
	vpbroadcastd	_ZZ4mainE3scp(%rip), %zmm31
	vmovdqa64	-1392(%rbp), %zmm28
	vmovdqa64	-1136(%rbp), %zmm27
	vmovdqa32	%zmm6, -1776(%rbp)
	vpmulld	8+_ZZ4mainE3mnp(%rip){1to16}, %zmm0, %zmm6
	vpbroadcastd	4+_ZZ4mainE3scp(%rip), %zmm29
	vmovdqa64	-1328(%rbp), %zmm26
	vmovdqa64	-1072(%rbp), %zmm25
	vmovdqa64	-1264(%rbp), %zmm24
	vmovdqa64	-1008(%rbp), %zmm23
	vmovdqa64	-1200(%rbp), %zmm22
	vmovdqa64	-944(%rbp), %zmm21
	vmovdqa32	%zmm6, -1904(%rbp)
	vpmulld	12+_ZZ4mainE3mnp(%rip){1to16}, %zmm0, %zmm6
	vmovdqa32	%zmm6, -2032(%rbp)
	.p2align 4
	.p2align 3
.L30:
	movzbl	%r13b, %edx
	vmovdqa32	%zmm8, %zmm3
	vmovdqa32	%zmm8, %zmm2
	vpbroadcastd	(%rax,%rdx,4), %zmm0
	leal	1(%r13), %edx
	vmovdqa32	%zmm8, %zmm1
	vpaddd	%zmm27, %zmm30, %zmm20
	movzbl	%dl, %edx
	vmovdqa64	%zmm20, %zmm27
	vpdpbusd	%zmm16, %zmm0, %zmm3
	vpbroadcastd	(%rax,%rdx,4), %zmm0
	leal	2(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm16, %zmm0, %zmm2
	vpbroadcastd	(%rax,%rdx,4), %zmm0
	leal	3(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm16, %zmm0, %zmm1
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	4(%r13), %edx
	vmovdqa32	%zmm8, %zmm0
	movzbl	%dl, %edx
	vpdpbusd	%zmm16, %zmm4, %zmm0
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	5(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm15, %zmm4, %zmm3
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	6(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm15, %zmm4, %zmm2
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	7(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm15, %zmm4, %zmm1
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	8(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm15, %zmm4, %zmm0
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	9(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm14, %zmm4, %zmm3
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	10(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm14, %zmm4, %zmm2
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	11(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm14, %zmm4, %zmm1
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	12(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm14, %zmm4, %zmm0
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	13(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm13, %zmm4, %zmm3
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	14(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm13, %zmm4, %zmm2
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	15(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm13, %zmm4, %zmm1
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	16(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm13, %zmm4, %zmm0
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	17(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm12, %zmm4, %zmm3
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	18(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm12, %zmm4, %zmm2
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	19(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm12, %zmm4, %zmm1
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	20(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm12, %zmm4, %zmm0
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	21(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm11, %zmm4, %zmm3
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	22(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm11, %zmm4, %zmm2
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	23(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm11, %zmm4, %zmm1
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	24(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm11, %zmm4, %zmm0
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	25(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm10, %zmm4, %zmm3
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	26(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm10, %zmm4, %zmm2
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	27(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm10, %zmm4, %zmm1
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	28(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm10, %zmm4, %zmm0
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	29(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm9, %zmm4, %zmm3
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	30(%r13), %edx
	movzbl	%dl, %edx
	vpdpbusd	%zmm9, %zmm4, %zmm2
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	leal	31(%r13), %edx
	addl	$32, %r13d
	movzbl	%dl, %edx
	vpdpbusd	%zmm9, %zmm4, %zmm1
	vpbroadcastd	(%rax,%rdx,4), %zmm4
	vpaddd	-1776(%rbp), %zmm25, %zmm19
	vpaddd	-1904(%rbp), %zmm23, %zmm18
	vpaddd	-2032(%rbp), %zmm21, %zmm17
	vpdpbusd	%zmm9, %zmm4, %zmm0
	vpmulld	%zmm31, %zmm3, %zmm7
	vpmulld	-1840(%rbp), %zmm1, %zmm5
	vpaddd	%zmm24, %zmm5, %zmm5
	vmovdqa64	%zmm19, %zmm25
	vmovdqa64	%zmm5, %zmm24
	vmovdqa64	%zmm18, %zmm23
	vpmulld	%zmm29, %zmm2, %zmm6
	vmovdqa64	%zmm17, %zmm21
	vpmulld	-1968(%rbp), %zmm0, %zmm4
	vpaddd	%zmm22, %zmm4, %zmm4
	vmovdqa64	%zmm4, %zmm22
	vpaddd	%zmm28, %zmm7, %zmm7
	vmovdqa64	%zmm7, %zmm28
	vpaddd	%zmm26, %zmm6, %zmm6
	vmovdqa64	%zmm6, %zmm26
	cmpl	%ebx, %r13d
	jne	.L30
	vmovdqa32	%zmm3, -880(%rbp)
	vmovdqa32	%zmm2, -816(%rbp)
	vmovdqa32	%zmm1, -752(%rbp)
	vmovdqa32	%zmm0, -688(%rbp)
	vmovdqa64	%zmm7, -1392(%rbp)
	vmovdqa64	%zmm20, -1136(%rbp)
	vmovdqa64	%zmm6, -1328(%rbp)
	vmovdqa64	%zmm19, -1072(%rbp)
	vmovdqa64	%zmm5, -1264(%rbp)
	vmovdqa64	%zmm18, -1008(%rbp)
	vmovdqa64	%zmm4, -1200(%rbp)
	vmovdqa64	%zmm17, -944(%rbp)
.L29:
#APP
# 32 "microbench-dpbusd.cpp" 1
	rdtsc
# 0 "" 2
#NO_APP
	vmovdqa32	-1328(%rbp), %zmm0
	salq	$32, %rdx
	movl	%eax, %eax
	movq	-1656(%rbp), %rdi
	vpaddd	-1392(%rbp), %zmm0, %zmm0
	orq	%rax, %rdx
	vpaddd	-816(%rbp), %zmm0, %zmm0
	vmovdqa32	-1264(%rbp), %zmm1
	subq	%rcx, %rdx
	vpaddd	-1008(%rbp), %zmm0, %zmm0
	vpaddd	-1200(%rbp), %zmm1, %zmm1
	movq	%rdx, %r12
	vpaddd	-1072(%rbp), %zmm1, %zmm1
	vpaddd	-688(%rbp), %zmm1, %zmm1
	vpaddd	%zmm1, %zmm0, %zmm0
	vmovdqa32	-880(%rbp), %zmm1
	vpaddd	-1136(%rbp), %zmm1, %zmm1
	vpaddd	-752(%rbp), %zmm1, %zmm1
	vpaddd	-944(%rbp), %zmm1, %zmm1
	vpaddd	%zmm1, %zmm0, %zmm0
	vextracti64x4	$0x1, %zmm0, %ymm1
	vpaddd	%ymm0, %ymm1, %ymm0
	vextracti64x2	$0x1, %ymm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm0
	vpshufd	$78, %xmm0, %xmm1
	vpaddd	%xmm0, %xmm1, %xmm1
	vmovdqa	%xmm1, %xmm0
	vpshufd	$85, %xmm1, %xmm1
	vpaddd	%xmm1, %xmm0, %xmm0
	vmovd	%xmm0, %esi
	movslq	%esi, %rsi
	movq	%rsi, _ZL6sink_v(%rip)
	vzeroupper
	call	_ZL8freq_khzPKc
	movl	%ebx, %edx
	leaq	-1432(%rbp), %rcx
	movl	%eax, %eax
	movq	%r12, %rsi
	leaq	.LC49(%rip), %rdi
	imulq	$274877907, %rax, %rax
	shrq	$38, %rax
	movl	%eax, -1400(%rbp)
	call	_ZL6reportPKcmiS0_P6result.isra.0
	movl	$2, %edi
	movl	$2, %eax
	vmovsd	-1608(%rbp), %xmm2
	vmovsd	-1616(%rbp), %xmm1
	leaq	.LC50(%rip), %rsi
	vmovsd	%xmm2, -1656(%rbp)
	vmovapd	%xmm2, %xmm0
	call	__printf_chk@PLT
	movl	$2, %edi
	movl	$1, %eax
	vmovsd	-1408(%rbp), %xmm1
	vmulsd	.LC51(%rip), %xmm1, %xmm0
	leaq	.LC52(%rip), %rsi
	vmovsd	-1656(%rbp), %xmm2
	vdivsd	%xmm2, %xmm0, %xmm0
	vcvttsd2sil	%xmm0, %edx
	vmovapd	%xmm1, %xmm0
	call	__printf_chk@PLT
	movl	$2, %edi
	movl	$1, %eax
	vmovsd	-1544(%rbp), %xmm1
	vsubsd	-1624(%rbp), %xmm1, %xmm0
	leaq	.LC53(%rip), %rsi
	vmovsd	%xmm1, -1656(%rbp)
	call	__printf_chk@PLT
	movl	$2, %edi
	movl	$1, %eax
	vmovsd	-1504(%rbp), %xmm0
	vmovsd	-1656(%rbp), %xmm1
	leaq	.LC54(%rip), %rsi
	vsubsd	%xmm1, %xmm0, %xmm0
	call	__printf_chk@PLT
	movl	$2, %edi
	movl	$1, %eax
	vmovsd	-1424(%rbp), %xmm0
	vsubsd	-1464(%rbp), %xmm0, %xmm0
	leaq	.LC55(%rip), %rsi
	call	__printf_chk@PLT
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L49
	addq	$1984, %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
.L47:
	.cfi_restore_state
	movq	16(%rsi), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	__isoc23_strtol@PLT
	movl	%eax, -1776(%rbp)
	jmp	.L14
.L48:
	vpxor	%xmm1, %xmm1, %xmm1
	vmovdqa64	%zmm1, %zmm10
	vmovdqa64	%zmm1, %zmm11
	vmovdqa64	%zmm1, %zmm0
	jmp	.L26
.L34:
	xorl	%eax, %eax
	jmp	.L16
.L49:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE6653:
	.size	main, .-main
	.local	_ZZ4mainE3mnp
	.comm	_ZZ4mainE3mnp,16,16
	.local	_ZZ4mainE3scp
	.comm	_ZZ4mainE3scp,16,16
	.local	_ZZ4mainE2ap
	.comm	_ZZ4mainE2ap,1024,64
	.local	_ZZ4mainE2bz
	.comm	_ZZ4mainE2bz,512,32
	.local	_ZL6sink_v
	.comm	_ZL6sink_v,8,8
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC3:
	.long	0
	.long	1080033280
	.align 8
.LC4:
	.long	0
	.long	1093567616
	.align 8
.LC5:
	.long	0
	.long	1083129856
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC8:
	.byte	3
	.byte	10
	.byte	17
	.byte	24
	.byte	31
	.byte	38
	.byte	45
	.byte	52
	.byte	59
	.byte	66
	.byte	73
	.byte	80
	.byte	87
	.byte	94
	.byte	101
	.byte	108
	.byte	115
	.byte	122
	.byte	-127
	.byte	-120
	.byte	-113
	.byte	-106
	.byte	-99
	.byte	-92
	.byte	-85
	.byte	-78
	.byte	-71
	.byte	-64
	.byte	-57
	.byte	-50
	.byte	-43
	.byte	-36
	.align 32
.LC9:
	.byte	-29
	.byte	-22
	.byte	-15
	.byte	-8
	.byte	-1
	.byte	6
	.byte	13
	.byte	20
	.byte	27
	.byte	34
	.byte	41
	.byte	48
	.byte	55
	.byte	62
	.byte	69
	.byte	76
	.byte	83
	.byte	90
	.byte	97
	.byte	104
	.byte	111
	.byte	118
	.byte	125
	.byte	-124
	.byte	-117
	.byte	-110
	.byte	-103
	.byte	-96
	.byte	-89
	.byte	-82
	.byte	-75
	.byte	-68
	.align 32
.LC10:
	.byte	-61
	.byte	-54
	.byte	-47
	.byte	-40
	.byte	-33
	.byte	-26
	.byte	-19
	.byte	-12
	.byte	-5
	.byte	2
	.byte	9
	.byte	16
	.byte	23
	.byte	30
	.byte	37
	.byte	44
	.byte	51
	.byte	58
	.byte	65
	.byte	72
	.byte	79
	.byte	86
	.byte	93
	.byte	100
	.byte	107
	.byte	114
	.byte	121
	.byte	-128
	.byte	-121
	.byte	-114
	.byte	-107
	.byte	-100
	.align 32
.LC11:
	.byte	-93
	.byte	-86
	.byte	-79
	.byte	-72
	.byte	-65
	.byte	-58
	.byte	-51
	.byte	-44
	.byte	-37
	.byte	-30
	.byte	-23
	.byte	-16
	.byte	-9
	.byte	-2
	.byte	5
	.byte	12
	.byte	19
	.byte	26
	.byte	33
	.byte	40
	.byte	47
	.byte	54
	.byte	61
	.byte	68
	.byte	75
	.byte	82
	.byte	89
	.byte	96
	.byte	103
	.byte	110
	.byte	117
	.byte	124
	.align 32
.LC12:
	.byte	-125
	.byte	-118
	.byte	-111
	.byte	-104
	.byte	-97
	.byte	-90
	.byte	-83
	.byte	-76
	.byte	-69
	.byte	-62
	.byte	-55
	.byte	-48
	.byte	-41
	.byte	-34
	.byte	-27
	.byte	-20
	.byte	-13
	.byte	-6
	.byte	1
	.byte	8
	.byte	15
	.byte	22
	.byte	29
	.byte	36
	.byte	43
	.byte	50
	.byte	57
	.byte	64
	.byte	71
	.byte	78
	.byte	85
	.byte	92
	.align 32
.LC13:
	.byte	99
	.byte	106
	.byte	113
	.byte	120
	.byte	127
	.byte	-122
	.byte	-115
	.byte	-108
	.byte	-101
	.byte	-94
	.byte	-87
	.byte	-80
	.byte	-73
	.byte	-66
	.byte	-59
	.byte	-52
	.byte	-45
	.byte	-38
	.byte	-31
	.byte	-24
	.byte	-17
	.byte	-10
	.byte	-3
	.byte	4
	.byte	11
	.byte	18
	.byte	25
	.byte	32
	.byte	39
	.byte	46
	.byte	53
	.byte	60
	.align 32
.LC14:
	.byte	67
	.byte	74
	.byte	81
	.byte	88
	.byte	95
	.byte	102
	.byte	109
	.byte	116
	.byte	123
	.byte	-126
	.byte	-119
	.byte	-112
	.byte	-105
	.byte	-98
	.byte	-91
	.byte	-84
	.byte	-77
	.byte	-70
	.byte	-63
	.byte	-56
	.byte	-49
	.byte	-42
	.byte	-35
	.byte	-28
	.byte	-21
	.byte	-14
	.byte	-7
	.byte	0
	.byte	7
	.byte	14
	.byte	21
	.byte	28
	.align 32
.LC15:
	.byte	35
	.byte	42
	.byte	49
	.byte	56
	.byte	63
	.byte	70
	.byte	77
	.byte	84
	.byte	91
	.byte	98
	.byte	105
	.byte	112
	.byte	119
	.byte	126
	.byte	-123
	.byte	-116
	.byte	-109
	.byte	-102
	.byte	-95
	.byte	-88
	.byte	-81
	.byte	-74
	.byte	-67
	.byte	-60
	.byte	-53
	.byte	-46
	.byte	-39
	.byte	-32
	.byte	-25
	.byte	-18
	.byte	-11
	.byte	-4
	.section	.rodata
	.align 64
.LC16:
	.long	0
	.long	-1640531535
	.long	1013904226
	.long	-626627309
	.long	2027808452
	.long	387276917
	.long	-1253254618
	.long	1401181143
	.long	-239350392
	.long	-1879881927
	.long	774553834
	.long	-865977701
	.long	1788458060
	.long	147926525
	.long	-1492605010
	.long	1161830751
	.align 64
.LC17:
	.long	-478700784
	.long	-2119232319
	.long	535203442
	.long	-1105328093
	.long	1549107668
	.long	-91423867
	.long	-1731955402
	.long	922480359
	.long	-718051176
	.long	1936384585
	.long	295853050
	.long	-1344678485
	.long	1309757276
	.long	-330774259
	.long	-1971305794
	.long	683129967
	.align 64
.LC18:
	.long	-957401568
	.long	1697034193
	.long	56502658
	.long	-1584028877
	.long	1070406884
	.long	-570124651
	.long	2084311110
	.long	443779575
	.long	-1196751960
	.long	1457683801
	.long	-182847734
	.long	-1823379269
	.long	831056492
	.long	-809475043
	.long	1844960718
	.long	204429183
	.align 64
.LC19:
	.long	-1436102352
	.long	1218333409
	.long	-422198126
	.long	-2062729661
	.long	591706100
	.long	-1048825435
	.long	1605610326
	.long	-34921209
	.long	-1675452744
	.long	978983017
	.long	-661548518
	.long	1992887243
	.long	352355708
	.long	-1288175827
	.long	1366259934
	.long	-274271601
	.align 64
.LC20:
	.long	-1914803136
	.long	739632625
	.long	-900898910
	.long	1753536851
	.long	113005316
	.long	-1527526219
	.long	1126909542
	.long	-513621993
	.long	2140813768
	.long	500282233
	.long	-1140249302
	.long	1514186459
	.long	-126345076
	.long	-1766876611
	.long	887559150
	.long	-752972385
	.align 64
.LC21:
	.long	1901463376
	.long	260931841
	.long	-1379599694
	.long	1274836067
	.long	-365695468
	.long	-2006227003
	.long	648208758
	.long	-992322777
	.long	1662112984
	.long	21581449
	.long	-1618950086
	.long	1035485675
	.long	-605045860
	.long	2049389901
	.long	408858366
	.long	-1231673169
	.align 64
.LC22:
	.long	1422762592
	.long	-217768943
	.long	-1858300478
	.long	796135283
	.long	-844396252
	.long	1810039509
	.long	169507974
	.long	-1471023561
	.long	1183412200
	.long	-457119335
	.long	-2097650870
	.long	556784891
	.long	-1083746644
	.long	1570689117
	.long	-69842418
	.long	-1710373953
	.align 64
.LC23:
	.long	944061808
	.long	-696469727
	.long	1957966034
	.long	317434499
	.long	-1323097036
	.long	1331338725
	.long	-309192810
	.long	-1949724345
	.long	704711416
	.long	-935820119
	.long	1718615642
	.long	78084107
	.long	-1562447428
	.long	1091988333
	.long	-548543202
	.long	2105892559
	.align 64
.LC24:
	.long	465361024
	.long	-1175170511
	.long	1479265250
	.long	-161266285
	.long	-1801797820
	.long	852637941
	.long	-787893594
	.long	1866542167
	.long	226010632
	.long	-1414520903
	.long	1239914858
	.long	-400616677
	.long	-2041148212
	.long	613287549
	.long	-1027243986
	.long	1627191775
	.align 64
.LC25:
	.long	-13339760
	.long	-1653871295
	.long	1000564466
	.long	-639967069
	.long	2014468692
	.long	373937157
	.long	-1266594378
	.long	1387841383
	.long	-252690152
	.long	-1893221687
	.long	761214074
	.long	-879317461
	.long	1775118300
	.long	134586765
	.long	-1505944770
	.long	1148490991
	.align 64
.LC26:
	.long	-492040544
	.long	-2132572079
	.long	521863682
	.long	-1118667853
	.long	1535767908
	.long	-104763627
	.long	-1745295162
	.long	909140599
	.long	-731390936
	.long	1923044825
	.long	282513290
	.long	-1358018245
	.long	1296417516
	.long	-344114019
	.long	-1984645554
	.long	669790207
	.align 64
.LC27:
	.long	-970741328
	.long	1683694433
	.long	43162898
	.long	-1597368637
	.long	1057067124
	.long	-583464411
	.long	2070971350
	.long	430439815
	.long	-1210091720
	.long	1444344041
	.long	-196187494
	.long	-1836719029
	.long	817716732
	.long	-822814803
	.long	1831620958
	.long	191089423
	.align 64
.LC28:
	.long	-1449442112
	.long	1204993649
	.long	-435537886
	.long	-2076069421
	.long	578366340
	.long	-1062165195
	.long	1592270566
	.long	-48260969
	.long	-1688792504
	.long	965643257
	.long	-674888278
	.long	1979547483
	.long	339015948
	.long	-1301515587
	.long	1352920174
	.long	-287611361
	.align 64
.LC29:
	.long	-1928142896
	.long	726292865
	.long	-914238670
	.long	1740197091
	.long	99665556
	.long	-1540865979
	.long	1113569782
	.long	-526961753
	.long	2127474008
	.long	486942473
	.long	-1153589062
	.long	1500846699
	.long	-139684836
	.long	-1780216371
	.long	874219390
	.long	-766312145
	.align 64
.LC30:
	.long	1888123616
	.long	247592081
	.long	-1392939454
	.long	1261496307
	.long	-379035228
	.long	-2019566763
	.long	634868998
	.long	-1005662537
	.long	1648773224
	.long	8241689
	.long	-1632289846
	.long	1022145915
	.long	-618385620
	.long	2036050141
	.long	395518606
	.long	-1245012929
	.align 64
.LC31:
	.long	1409422832
	.long	-231108703
	.long	-1871640238
	.long	782795523
	.long	-857736012
	.long	1796699749
	.long	156168214
	.long	-1484363321
	.long	1170072440
	.long	-470459095
	.long	-2110990630
	.long	543445131
	.long	-1097086404
	.long	1557349357
	.long	-83182178
	.long	-1723713713
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC32:
	.long	3
	.long	-7
	.long	12
	.long	-1
	.align 16
.LC33:
	.long	2
	.long	-3
	.long	5
	.long	0
	.section	.rodata.cst8
	.align 8
.LC51:
	.long	0
	.long	1079574528
	.ident	"GCC: (Ubuntu 14.2.0-19ubuntu2) 14.2.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
