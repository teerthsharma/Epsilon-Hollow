use aether_core::memory::*;

fn noop<T>(_h: &mut ManifoldHeap<T>) {}

#[test]
fn probe_generation_never_increments() {
    let mut heap = ManifoldHeap::<u64>::new();
    let mut hs = Vec::new();
    for i in 0..16u64 {
        hs.push(heap.alloc(i));
    }
    // heat everything except slot 0
    for h in hs.iter().skip(1) {
        for _ in 0..8 {
            heap.touch(*h);
        }
    }
    let cold = hs[0];
    println!(
        "gens: {:?}",
        hs.iter().map(|h| h.generation()).collect::<Vec<_>>()
    );
    let pruned = heap.regulate_entropy(noop);
    println!("pruned={} active={}", pruned, heap.active_count());
    println!("cold still readable after prune: {:?}", heap.get(cold));
    let fresh = heap.alloc(999);
    println!(
        "fresh idx={} gen={} (cold idx={} gen={})",
        fresh.index(),
        fresh.generation(),
        cold.index(),
        cold.generation()
    );
    println!("STALE HANDLE READ: {:?}", heap.get(cold));
}

#[test]
fn probe_all_garbage_collection_fraction() {
    for n in [8usize, 64, 256] {
        let mut heap = ManifoldHeap::<u64>::new();
        for i in 0..n {
            heap.alloc(i as u64);
        }
        // every object is unreachable: tracer marks nothing
        let mut total = 0;
        let mut passes = 0;
        for _ in 0..50 {
            let p = heap.regulate_entropy(noop);
            total += p;
            passes += 1;
            if p == 0 {
                break;
            }
        }
        println!(
            "n={} uniform-liveness all-garbage: reclaimed {}/{} over {} passes, active={}",
            n,
            total,
            n,
            passes,
            heap.active_count()
        );
    }
}

#[test]
fn probe_chebyshev_cap_with_spread() {
    let n = 256usize;
    let mut heap = ManifoldHeap::<u64>::new();
    let mut hs = Vec::new();
    for i in 0..n {
        hs.push(heap.alloc(i as u64));
    }
    // give a spread of liveness: object i touched i%20 times
    for (i, h) in hs.iter().enumerate() {
        for _ in 0..(i % 20) {
            heap.touch(*h);
        }
    }
    let mut alive = n;
    for pass in 0..12 {
        let p = heap.regulate_entropy(noop);
        println!(
            "pass {}: pruned {} of {} alive ({:.1}%)",
            pass,
            p,
            alive,
            100.0 * p as f64 / alive as f64
        );
        alive -= p;
        if p == 0 {
            break;
        }
    }
    println!(
        "final active={} of {} (all were garbage)",
        heap.active_count(),
        n
    );
}

#[test]
fn probe_liveness_ceiling() {
    let mut heap = ManifoldHeap::<u64>::new();
    let h = heap.alloc(1);
    for _ in 0..100 {
        heap.touch(h);
        heap.mark(h);
        heap.get_mut(h);
    }
    println!(
        "liveness after 100 touch+mark+get_mut: {}",
        heap.blocks[0].liveness[0]
    );
}

#[test]
fn probe_tree_stats_ever_updated() {
    let mut heap = ManifoldHeap::<u64>::new();
    for i in 0..64u64 {
        let h = heap.alloc(i);
        for _ in 0..5 {
            heap.touch(h);
        }
    }
    heap.regulate_entropy(noop);
    for (i, node) in heap.nodes.iter().enumerate() {
        println!(
            "node {}: mean_liveness={} max_liveness={} leaf_parent={}",
            i, node.mean_liveness, node.max_liveness, node.is_leaf_parent
        );
    }
    println!("blocks={} nodes={}", heap.blocks.len(), heap.nodes.len());
}

#[test]
fn probe_marked_object_is_never_freed() {
    let mut heap = ManifoldHeap::<u64>::new();
    let mut hs = Vec::new();
    for i in 0..64u64 {
        hs.push(heap.alloc(i));
    }
    let keep = hs[0];
    for _ in 0..20 {
        heap.regulate_entropy(|h| {
            h.mark(keep);
        });
    }
    println!(
        "marked survivor readable: {:?}, active={}",
        heap.get(keep),
        heap.active_count()
    );
}
