/* Seal OS Linker Script — places Multiboot2 header at the start */
ENTRY(_start)

SECTIONS {
    . = 1M;

    .multiboot_header : ALIGN(8) {
        KEEP(*(.multiboot_header))
    }

    .text : ALIGN(4K) {
        *(.text .text.*)
    }

    .rodata : ALIGN(4K) {
        *(.rodata .rodata.*)
    }

    .data : ALIGN(4K) {
        *(.data .data.*)
    }

    .bss : ALIGN(4K) {
        *(COMMON)
        *(.bss .bss.*)
    }

    /DISCARD/ : {
        *(.eh_frame)
        *(.note.GNU-stack)
    }
}
