lines = [
"[GPU-BENCH] kernel=voronoi_assign mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_points=64 n_cells=8 warmup=4 iterations=32 dispatch_ok=36 wait_ok=36 upload_ok=2 download_ok=1 verify=pass reference=cpu_recompute checked=64 mismatches=0 invalid_ids=0 checksum=15967573174722442626 first_cell=0 avg_cycles=24679371",
"[GPU-BENCH] kernel=jl_project mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_vectors=4 dim_in=128 dim_out=3 seed=0xE95110A7 warmup=4 iterations=32 dispatch_ok=36 wait_ok=36 upload_ok=1 download_ok=1 verify=pass reference=cpu_recompute checked=12 mismatches=0 finite=12 checksum=15986779674487761376 max_abs_diff_scaled=0 avg_cycles=973648",
"[GPU-BENCH] kernel=spectral_step mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 dim=512 alpha_ppm=300000 warmup=4 iterations=32 dispatch_ok=36 wait_ok=36 upload_ok=1 download_ok=1 verify=pass reference=cpu_recompute checked=512 mismatches=0 finite=512 checksum=5411236274378254326 max_abs_diff_scaled=0 avg_cycles=83437"
]

kernel_lines = sum(1 for line in lines if line.startswith("[GPU-BENCH] kernel=") and "mode=" in line)
print(kernel_lines)
