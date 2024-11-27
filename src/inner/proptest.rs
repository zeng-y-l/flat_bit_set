use crate::*;
use core::{fmt::Debug, marker::PhantomData};
use prop::{collection::vec, strategy::ValueTree};
use proptest::prelude::*;
use rand::{distributions::Standard, prelude::*};

struct ChunkValueTree<C> {
    cur: C,
    prev: Option<C>,
}

#[derive(Clone, Debug)]
struct ChunkStrategy<C>(PhantomData<C>);

impl<C> Strategy for ChunkStrategy<C>
where
    C: Chunk + Debug,
    Standard: Distribution<C>,
{
    type Tree = ChunkValueTree<C>;

    type Value = C;

    fn new_tree(
        &self,
        runner: &mut prop::test_runner::TestRunner,
    ) -> prop::strategy::NewTree<Self> {
        Ok(ChunkValueTree {
            cur: runner.rng().gen(),
            prev: None,
        })
    }
}

impl<C: Chunk + Debug> ValueTree for ChunkValueTree<C> {
    type Value = C;

    fn current(&self) -> Self::Value {
        self.cur
    }

    fn simplify(&mut self) -> bool {
        let ok = self.cur.count_ones() > 1;
        if ok {
            self.prev = Some(self.cur);
            self.cur.remove_lowest_bit();
        }
        ok
    }

    fn complicate(&mut self) -> bool {
        self.prev.take().is_some_and(|prev| {
            self.cur = prev;
            true
        })
    }
}

impl<K, C> Arbitrary for FlatBitSet<K, C>
where
    K: Key + Debug + Arbitrary,
    C: Chunk + Debug,
    Standard: Distribution<C>,
{
    type Parameters = prop::sample::SizeRange;

    fn arbitrary_with(size: Self::Parameters) -> Self::Strategy {
        let ele = (
            any::<K>().prop_map(|k| k.split::<C>().0),
            ChunkStrategy::<C>(PhantomData),
        );
        let st = vec(ele, size).prop_map(|mut vec| {
            vec.sort_unstable_by_key(|&(k, _)| k);
            vec.dedup_by_key(|&mut (k, _)| k);
            for (_, c) in &vec {
                debug_assert!(c.count_ones() != 0);
            }
            Self(vec)
        });
        st.boxed()
    }

    type Strategy = BoxedStrategy<Self>;
}
