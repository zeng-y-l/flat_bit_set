mod state;

use crate::*;
use biop::*;
use core::{assert_eq, cmp::Ordering, fmt::Debug, u32};
use proptest::prelude::*;
use std::{collections::BTreeSet, usize};

type Res = Result<(), TestCaseError>;

#[test]
fn test_key() {
    fn check<K: Key + Debug, C: Chunk>(k1: K, k2: K) -> Res {
        let (k1_, b1) = k1.split::<C>();
        prop_assert!(b1 < C::BITS);
        prop_assert_eq!(k1, k1_.merge::<C>(b1));
        let (k2_, b2) = k2.split::<C>();
        prop_assert_eq!(k1.cmp(&k2), (k1_, b1).cmp(&(k2_, b2)));
        Ok(())
    }

    fn test<K: Key + Debug + Arbitrary>() {
        proptest!(|(k1: K, k2: K)| {
            check::<_, u32>(k1, k2)?;
            check::<_, u64>(k1, k2)?;
            check::<_, u128>(k1, k2)?;
        });
    }

    test::<u8>();
    test::<i8>();
    test::<u128>();
    test::<i128>();
    test::<usize>();
    test::<isize>();
}

#[test]
fn test_chunk() {
    fn check<C: Chunk + Debug + Arbitrary>() {
        assert_eq!(C::ZERO.count_ones(), 0);
        assert_eq!((!C::ZERO).count_ones(), C::BITS);

        for bit in 0..C::BITS {
            let mut c = C::mask(bit);
            assert_eq!(c.count_ones(), 1);
            assert_eq!(c.highest_bit(), bit);
            assert_eq!(c.lowest_bit(), bit);
            c.remove_lowest_bit();
            assert_eq!(c, C::ZERO);
        }

        proptest!(|(c: C)| {
            let ones = c.count_ones();
            if ones == 0 {
                prop_assert_eq!(c, C::ZERO);
                return Ok(());
            }
            let bit = c.lowest_bit();
            let mut c2 = c;
            c2.remove_lowest_bit();
            prop_assert_eq!(c2.count_ones(), ones - 1);
            prop_assert_eq!(c2 | C::mask(bit), c);
        });
    }

    check::<u32>();
    check::<u64>();
    check::<u128>();
    check::<usize>();
}

#[test]
fn test_iter() {
    fn check<K: Key + Debug, C: Chunk + Debug>(set: &FlatBitSet<K, C>, nxt: &[usize]) -> Res {
        let mut iter = set.iter();
        let (lo, hi) = iter.size_hint();
        let count = iter.by_ref().count();
        prop_assert_eq!(count, set.iter().map(|_| 1).sum::<usize>());
        prop_assert!(lo <= count && count <= hi.unwrap_or(usize::MAX));
        prop_assert!(iter.next().is_none());

        let last = set.iter().try_fold(None, |prev, cur| {
            if let Some(prev) = prev {
                prop_assert!(prev < cur);
            }
            Ok(Some(cur))
        })?;
        prop_assert_eq!(last, set.iter().last());

        prop_assert_eq!(&set.iter().collect::<FlatBitSet<K, C>>(), set);

        let mut iter = set.iter();
        for &n in nxt {
            let mut iter2 = iter.clone();
            prop_assert_eq!(iter.nth(n), std::iter::from_fn(|| iter2.next()).nth(n));
        }

        Ok(())
    }

    fn test<K: Key + Debug, C: Chunk + Debug>()
    where
        FlatBitSet<K, C>: Arbitrary,
    {
        let s = any::<FlatBitSet<K, C>>().prop_flat_map(|set| {
            let nxt = prop::collection::vec(0..set.count() + 5, 0..10).prop_map(|mut nxt| {
                nxt.sort_unstable();
                nxt.iter_mut().reduce(|prev, cur| {
                    *cur -= *prev;
                    cur
                });
                nxt
            });
            (Just(set), nxt)
        });
        proptest!(|((set, nxt) in s)| check(&set, &nxt)?);
    }

    test::<u8, u32>();
    test::<i16, u128>();
    test::<u32, u64>();
    test::<i64, usize>();
    test::<u128, u32>();
}

#[test]
fn test_biop() {
    fn check<O: BiOp<C>, C: Chunk + Debug>(c: C) -> Res {
        prop_assert_eq!(O::first(c), O::both(c, C::ZERO));
        prop_assert_eq!(O::second(c), O::both(C::ZERO, c));
        Ok(())
    }

    fn test<O: BiOp<u32> + BiOp<u64> + BiOp<u128> + BiOp<usize>>() {
        proptest!(|(c: u32)| check::<O, _>(c)?);
        proptest!(|(c: u64)| check::<O, _>(c)?);
        proptest!(|(c: u128)| check::<O, _>(c)?);
        proptest!(|(c: usize)| check::<O, _>(c)?);
    }

    test::<AndOp>();
    test::<OrOp>();
    test::<XorOp>();
    test::<DiffOp>();
}

#[test]
fn test_set_biop() {
    fn check<K: Key + Debug, C: Chunk + Debug>(
        a: &FlatBitSet<K, C>,
        b: &FlatBitSet<K, C>,
        mut c: impl Iterator<Item = K> + Clone,
        d: impl Iterator<Item = K>,
    ) -> Res {
        prop_assert_eq!(&a, &b);
        prop_assert!(c.clone().eq(d));
        prop_assert!(a.iter().eq(c.clone()));
        let (lo, hi) = c.size_hint();
        let count = c.by_ref().count();
        prop_assert!(lo <= count);
        prop_assert!(count <= hi.unwrap_or(usize::MAX));
        prop_assert!(c.next().is_none());
        Ok(())
    }

    fn checks<K: Key + Debug, C: Chunk + Debug>(
        s1: &FlatBitSet<K, C>,
        s2: &FlatBitSet<K, C>,
    ) -> Res {
        let ts1 = s1.iter().collect::<BTreeSet<_>>();
        let ts2 = s2.iter().collect::<BTreeSet<_>>();

        check(
            &(s1 & s2),
            &(s1.clone() & s2),
            s1.intersection_iter(s2),
            ts1.intersection(&ts2).copied(),
        )?;
        check(
            &(s1 | s2),
            &(s1.clone() | s2),
            s1.union_iter(s2),
            ts1.union(&ts2).copied(),
        )?;
        check(
            &(s1 - s2),
            &(s1.clone() - s2),
            s1.difference_iter(s2),
            ts1.difference(&ts2).copied(),
        )?;
        check(
            &(s1 ^ s2),
            &(s1.clone() ^ s2),
            s1.symmetric_difference_iter(s2),
            ts1.symmetric_difference(&ts2).copied(),
        )?;

        Ok(())
    }

    fn test<K: Key + Debug, C: Chunk + Debug>()
    where
        FlatBitSet<K, C>: Arbitrary,
    {
        proptest!(|(s1: FlatBitSet<K, C>, s2: FlatBitSet<K, C>)| checks(&s1, &s2)?);
    }

    test::<i128, u32>();
    test::<usize, u64>();
    test::<i8, u128>();
}

#[test]
fn test_disjoint() {
    fn check<K: Key + Debug, C: Chunk + Debug>(a: &FlatBitSet<K, C>, b: &FlatBitSet<K, C>) -> Res {
        prop_assert_eq!(a.is_disjoint(b), a.intersection_iter(b).next().is_none());
        Ok(())
    }

    fn test<K: Key + Debug, C: Chunk + Debug>()
    where
        FlatBitSet<K, C>: Arbitrary,
    {
        proptest!(|(s1: FlatBitSet<K, C>, s2: FlatBitSet<K, C>)| check(&s1, &s2)?);
        proptest!(|(s1: FlatBitSet<K, C>, s2: FlatBitSet<K, C>)| check(&s1, &(&s1 | &s2))?);
    }

    test::<i128, u32>();
    test::<usize, u64>();
    test::<i8, u128>();
}

#[test]
fn test_subset() {
    fn check<K: Key + Debug, C: Chunk + Debug>(a: &FlatBitSet<K, C>, b: &FlatBitSet<K, C>) -> Res {
        let ord = match (a.is_subset(b), a.is_superset(b)) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        };
        prop_assert_eq!(a == b, ord == Some(Ordering::Equal));
        prop_assert_eq!(a.subset_cmp(b), ord);
        Ok(())
    }

    fn test<K: Key + Debug, C: Chunk + Debug>()
    where
        FlatBitSet<K, C>: Arbitrary,
    {
        proptest!(|(s1: FlatBitSet<K, C>, s2: FlatBitSet<K, C>)| check(&s1, &s2)?);
        proptest!(|(s1: FlatBitSet<K, C>, s2: FlatBitSet<K, C>)| check(&s1, &(&s1 | &s2))?);
    }

    test::<i128, u32>();
    test::<usize, u64>();
    test::<i8, u128>();
}

#[test]
fn test_serde() {
    fn test<K: Key, C: Chunk>()
    where
        FlatBitSet<K, C>: Arbitrary + Debug + serde::Serialize + serde::de::DeserializeOwned,
    {
        proptest!(|(set: FlatBitSet<K, C>)| {
            let json = serde_json::to_string(&set).unwrap();
            prop_assert_eq!(&set, &serde_json::from_str::<FlatBitSet<K, C>>(&json).unwrap());
        });
    }

    test::<u8, u32>();
    test::<i16, u128>();
    test::<u32, u64>();
    test::<i64, usize>();
    test::<u128, u32>();
}
