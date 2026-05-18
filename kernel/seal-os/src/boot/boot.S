/* Seal OS — 32-bit to 64-bit trampoline
 * GRUB multiboot2 enters here in 32-bit protected mode.
 * EAX = multiboot2 magic (0x36d76289)
 * EBX = physical address of multiboot2 info structure
 *
 * We set up identity-mapped page tables, enable long mode, and jump to _start(ebx).
 */

.section .multiboot_boot, "ax"
.code32
.global _boot
_boot:
    cli
    mov %ebx, %esi          /* save multiboot info pointer */

    /* Set up stack */
    lea boot_stack_top, %esp

    /* --- Set up page tables (identity map first 2GB using 2MB pages) --- */
    /* Zero out page tables */
    lea pml4, %edi
    xor %eax, %eax
    mov $6144, %ecx          /* 6 pages (pml4+pdpt+pd0-3) * 4096 bytes / 4 = 6144 dwords */
    rep stosl

    /* PML4[0] -> PDPT */
    lea pml4, %edi
    lea pdpt, %eax
    or $0x3, %eax            /* present + writable */
    mov %eax, (%edi)

    /* PDPT[0..3] -> PD0..PD3 (identity map first 4GB for MMIO framebuffers) */
    lea pdpt, %edi
    lea pd0, %eax
    or $0x3, %eax
    mov %eax, (%edi)
    lea pd1, %eax
    or $0x3, %eax
    mov %eax, 8(%edi)
    lea pd2, %eax
    or $0x3, %eax
    mov %eax, 16(%edi)
    lea pd3, %eax
    or $0x3, %eax
    mov %eax, 24(%edi)

    /* PD0..PD3: 4×512 entries, each mapping 2MB (covers 4GB) */
    lea pd0, %edi
    mov $0x83, %eax          /* present + writable + huge (2MB page) */
    mov $2048, %ecx          /* 4 × 512 entries */
1:
    mov %eax, (%edi)
    add $0x200000, %eax      /* next 2MB */
    add $8, %edi
    loop 1b

    /* --- Enable PAE --- */
    mov %cr4, %eax
    or $0x20, %eax           /* CR4.PAE */
    mov %eax, %cr4

    /* --- Load PML4 into CR3 --- */
    lea pml4, %eax
    mov %eax, %cr3

    /* --- Enable long mode via EFER MSR --- */
    mov $0xC0000080, %ecx    /* IA32_EFER */
    rdmsr
    or $0x100, %eax          /* EFER.LME (Long Mode Enable) */
    wrmsr

    /* --- Enable paging (activates long mode) --- */
    mov %cr0, %eax
    or $0x80000001, %eax     /* CR0.PG + CR0.PE */
    mov %eax, %cr0

    /* --- Load 64-bit GDT and far-jump to 64-bit code --- */
    lgdt gdt64_ptr
    ljmp $0x08, $long_mode_entry

.code64
long_mode_entry:
    /* Set up 64-bit segment registers */
    mov $0x10, %ax
    mov %ax, %ds
    mov %ax, %es
    mov %ax, %ss
    mov %ax, %fs
    mov %ax, %gs

    /* Set up 64-bit stack */
    lea boot_stack_top, %rsp

    /* Pass multiboot info pointer as first argument (rdi) */
    movl %esi, %edi
    xor %rsi, %rsi

    /* Jump to Rust _start */
    call _start

    /* Should never return */
2:
    hlt
    jmp 2b

/* --- Data --- */

.section .bss
.align 4096
pml4:   .skip 4096
pdpt:   .skip 4096
pd0:    .skip 4096
pd1:    .skip 4096
pd2:    .skip 4096
pd3:    .skip 4096

.align 16
boot_stack_bottom:
    .skip 65536              /* 64KB boot stack */
boot_stack_top:

.section .rodata
.align 8
gdt64:
    .quad 0                  /* null descriptor */
    .quad 0x00AF9A000000FFFF /* 64-bit code: base=0, limit=max, exec/read, long mode */
    .quad 0x00CF92000000FFFF /* 64-bit data: base=0, limit=max, read/write */
gdt64_ptr:
    .word gdt64_ptr - gdt64 - 1
    .quad gdt64
