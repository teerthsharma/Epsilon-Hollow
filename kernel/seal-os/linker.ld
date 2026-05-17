/* Seal OS Linker Script — kernel loaded at 1MB by GRUB */
ENTRY(_boot)

SECTIONS {
    . = 1M;

    .multiboot_header : ALIGN(8) {
        KEEP(*(.multiboot_header))
    }

    .multiboot_boot : ALIGN(4K) {
        KEEP(*(.multiboot_boot))
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
