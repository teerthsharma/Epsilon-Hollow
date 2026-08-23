//! Does the net-tree level profile recover box-counting dimension?
//! N(r) = |net at radius r|, dim = lim log N(r) / log(1/r).
use aether_core::nettree::NetTree;
use aether_core::manifold::ManifoldPoint;

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || { s^=s<<13; s^=s>>7; s^=s<<17; ((s>>11) as f64)/((1u64<<53) as f64) }
}

/// Local slope of log N vs log(1/r) between consecutive levels.
fn dimension_profile<const D: usize>(pts: &[ManifoldPoint<D>], label: &str, expect: f64) {
    let t = NetTree::build(pts, 2.0);
    println!("--- {label}  (n={}, expected dim ~= {expect}) ---", pts.len());
    let mut slopes = Vec::new();
    for l in 0..t.levels().saturating_sub(1) {
        let (n0, n1) = (t.net(l).len() as f64, t.net(l+1).len() as f64);
        let (r0, r1) = (t.radius(l), t.radius(l+1));
        if n1 < 2.0 || n0 < 2.0 { continue; }
        let slope = (n0/n1).ln() / (r1/r0).ln();
        println!("  r {:.5}->{:.5}   N {:4}->{:4}   local dim = {:.3}", r0, r1, n0 as usize, n1 as usize, slope);
        slopes.push(slope);
    }
    if slopes.len() >= 3 {
        // Drop the first and last: boundary effects at the finest and coarsest scale.
        let mid = &slopes[1..slopes.len()-1];
        let mean: f64 = mid.iter().sum::<f64>() / mid.len() as f64;
        println!("  ==> mid-scale mean dimension = {:.3}   (expected {expect})", mean);
    }
}

#[test]
fn box_counting_dimension_from_the_net_tree() {
    // 1-D: circle in R^3
    let n = 2000;
    let circle: Vec<ManifoldPoint<3>> = (0..n).map(|i| {
        let t = 2.0*core::f64::consts::PI*(i as f64)/(n as f64);
        ManifoldPoint::new([t.cos(), t.sin(), 0.0])
    }).collect();
    dimension_profile(&circle, "circle (1-manifold)", 1.0);

    // 2-D: uniform sphere sample
    let mut u = rng(12345);
    let sph: Vec<ManifoldPoint<3>> = (0..3000).map(|_| {
        let z = 2.0*u() - 1.0; let p = 2.0*core::f64::consts::PI*u();
        let r = (1.0 - z*z).max(0.0).sqrt();
        ManifoldPoint::new([r*p.cos(), r*p.sin(), z])
    }).collect();
    dimension_profile(&sph, "sphere S2 (2-manifold)", 2.0);

    // 3-D: uniform ball
    let mut u3 = rng(999);
    let ball: Vec<ManifoldPoint<3>> = (0..4000).map(|_| {
        loop {
            let (x,y,z) = (2.0*u3()-1.0, 2.0*u3()-1.0, 2.0*u3()-1.0);
            if x*x+y*y+z*z <= 1.0 { return ManifoldPoint::new([x,y,z]); }
        }
    }).collect();
    dimension_profile(&ball, "solid ball (3-manifold)", 3.0);

    // FRACTAL: Cantor dust in the plane, dim = log4/log3 = 1.2619
    let mut pts = vec![[0.0f64, 0.0f64]];
    for _ in 0..6 {
        let s = 1.0/3.0;
        let mut next = Vec::new();
        for p in &pts {
            for (dx,dy) in [(0.0,0.0),(2.0,0.0),(0.0,2.0),(2.0,2.0)] {
                next.push([p[0]*s + dx*s, p[1]*s + dy*s]);
            }
        }
        pts = next;
        if pts.len() > 4096 { break; }
    }
    let cantor: Vec<ManifoldPoint<3>> = pts.iter().map(|p| ManifoldPoint::new([p[0],p[1],0.0])).collect();
    dimension_profile(&cantor, "Cantor dust (log4/log3)", 1.2619);
}
