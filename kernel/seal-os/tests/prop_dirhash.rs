use proptest::prelude::*;
use seal_os::fs::dir_hash::DirHash;

proptest! {
    #[test]
    fn test_dirhash_invariant(names in prop::collection::vec("[a-zA-Z0-9]{1,20}", 1..100)) {
        let mut dh = DirHash::new(0);
        let parent_id = 1;
        for (i, name) in names.iter().enumerate() {
            dh.insert(parent_id, name, i as u64);
        }
        
        // Invariant: every inserted name is findable
        for name in &names {
            prop_assert!(dh.lookup(parent_id, name).is_some());
        }
    }
}
