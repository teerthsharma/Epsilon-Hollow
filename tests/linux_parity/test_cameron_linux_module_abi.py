"""What "use Linux drivers" costs when the drivers are Linux binaries.

Fixture, measured on Linux 6.6.87.2-microsoft-standard-WSL2 (Ubuntu 26.04):
`nm -u .../drivers/net/ethernet/intel/e1000/e1000.ko` lists the 152 symbols
below. modinfo reports `license: GPL v2` and
`vermagic: 6.6.87.2-microsoft-standard-WSL2 SMP preempt mod_unload modversions`;
the module carries a `__versions` CRC section, so it loads only into a kernel
exporting those symbols with matching CRCs. Across all 924 .ko files in that
kernel's /lib/modules the union of undefined symbols is 10,766.

e1000 is chosen because Seal OS already ships its own e1000 driver
(kernel/seal-os/src/drivers/net/e1000.rs), so the device is not the obstacle;
the Linux-internal interface is.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "kernel/seal-os/src"

E1000_KO_IMPORTS = """
__SCT__cond_resched ___pskb_trim __alloc_skb __const_udelay __dynamic_netdev_dbg __fentry__ __folio_put __kmalloc
__local_bh_enable_ip __napi_alloc_skb __napi_schedule __netdev_alloc_frag_align __pci_register_driver __pskb_pull_tail
__put_devmap_managed_page_refs __skb_pad __stack_chk_fail __this_cpu_preempt_check __udelay __warn_printk
__x86_indirect_thunk_r8 __x86_indirect_thunk_rax __x86_return_thunk _dev_err _dev_info _dev_warn _find_next_bit
_printk _raw_spin_lock _raw_spin_lock_irqsave _raw_spin_unlock _raw_spin_unlock_irqrestore alloc_etherdev_mqs
alloc_pages cancel_delayed_work_sync cancel_work_sync consume_skb csum_ipv6_magic debug_smp_processor_id
delayed_work_timer_fn dev_addr_mod dev_driver_string dev_kfree_skb_any_reason device_set_wakeup_enable
devmap_managed_key disable_hardirq dma_alloc_attrs dma_free_attrs dma_map_page_attrs dma_set_coherent_mask
dma_set_mask dma_sync_single_for_cpu dma_sync_single_for_device dma_unmap_page_attrs dql_completed dql_reset
enable_irq eth_type_trans eth_validate_addr ethtool_convert_legacy_u32_to_link_mode
ethtool_convert_link_mode_to_legacy_u32 ethtool_op_get_ts_info fortify_panic free_irq free_netdev
hugetlb_optimize_vmemmap_key init_timer_key ioread16_rep ioremap iounmap iowrite16_rep is_vmalloc_addr jiffies kfree
kfree_skb_reason kmalloc_caches kmalloc_trace memcpy memset msleep msleep_interruptible mutex_lock mutex_unlock
napi_build_skb napi_complete_done napi_consume_skb napi_disable napi_enable napi_get_frags napi_gro_frags
napi_gro_receive napi_schedule_prep net_ratelimit netdev_err netdev_info netdev_warn netif_carrier_off
netif_carrier_on netif_device_attach netif_device_detach netif_napi_add_weight netif_schedule_queue
netif_tx_wake_queue page_frag_free page_offset_base param_array_ops param_ops_int param_ops_uint pci_clear_mwi
pci_disable_device pci_enable_device pci_enable_device_mem pci_enable_wake pci_ioremap_bar pci_read_config_word
pci_release_selected_regions pci_request_selected_regions pci_save_state pci_select_bars pci_set_master pci_set_mwi
pci_set_power_state pci_unregister_driver pci_wake_from_d3 pcix_get_mmrbc pcix_set_mmrbc phys_base preempt_count_add
print_hex_dump pskb_expand_head queue_delayed_work_on queue_work_on register_netdev request_threaded_irq
skb_clone_tx_timestamp skb_copy_bits skb_put skb_trim skb_tstamp_tx softnet_data strchr strncpy strnlen strscpy
synchronize_irq system_state system_wq unregister_netdev usleep_range_state vfree vmemmap_base vzalloc
""".split()


def kernel_exported_symbols():
    """Names Seal exports with C linkage: #[no_mangle] fns/statics and #[export_name]."""
    names = set()
    for path in SRC.rglob("*.rs"):
        text = path.read_text(encoding="utf-8", errors="replace")
        names.update(re.findall(r"#\[no_mangle\]\s*(?:#\[[^\]]*\]\s*)*pub\s+(?:unsafe\s+)?(?:extern\s+\"C\"\s+)?(?:fn|static(?:\s+mut)?)\s+(\w+)", text))
        names.update(re.findall(r'#\[export_name\s*=\s*"(\w+)"\]', text))
    return names


def test_kernel_exports_what_a_linux_nic_driver_imports():
    assert len(E1000_KO_IMPORTS) == 152
    # memcpy/memset reach the kernel link from compiler_builtins, not from Seal source.
    exported = kernel_exported_symbols() | {"memcpy", "memset"}
    provided = sorted(set(E1000_KO_IMPORTS) & exported)
    assert len(provided) == len(E1000_KO_IMPORTS), (
        f"Seal exports {len(provided)}/152 symbols e1000.ko imports (provided: {provided}); "
        f"Seal's C-linkage exports: {len(exported)}"
    )
