use crate::*;

use core::iter::FusedIterator;

/// 通用迭代器
///
/// 请使用 [`Iter`] 或 [`IntoIter`]。
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Debug, Clone)]
pub struct GenericIter<I, K, C> {
    inner: I,
    key: K,
    cur: C,
}

impl<I, K: Key, C: Chunk> GenericIter<I, K, C> {
    pub(crate) fn new(inner: I) -> Self {
        Self {
            inner,
            key: K::default(),
            cur: C::ZERO,
        }
    }
}

impl<I, K, C> Iterator for GenericIter<I, K, C>
where
    I: Iterator<Item = (K, C)>,
    K: Key,
    C: Chunk,
{
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == C::ZERO {
            (self.key, self.cur) = self.inner.next()?;
        }
        debug_assert!(self.cur != C::ZERO);
        let cur = self.cur.lowest_bit();
        self.cur.remove_lowest_bit();
        Some(self.key.merge::<C>(cur))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.inner.size_hint();
        let cur = self.cur.count_ones() as usize;
        (
            lo.saturating_add(cur),
            hi.and_then(|x| x.checked_mul(C::BITS as usize)?.checked_add(cur)),
        )
    }

    fn count(self) -> usize {
        self.inner
            .map(|chunk| chunk.1.count_ones() as usize)
            .sum::<usize>()
            + self.cur.count_ones() as usize
    }

    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        while n >= self.cur.count_ones() as usize {
            n -= self.cur.count_ones() as usize;
            (self.key, self.cur) = self.inner.next()?;
        }
        for _ in 0..n {
            self.next()?;
        }
        self.next()
    }

    fn min(mut self) -> Option<Self::Item> {
        self.next()
    }

    fn last(self) -> Option<Self::Item> {
        let (k, c) = self.inner.last().unwrap_or((self.key, self.cur));
        (c != C::ZERO).then(|| k.merge::<C>(c.highest_bit()))
    }

    fn max(self) -> Option<Self::Item> {
        self.last()
    }
}

impl<I, K, C> FusedIterator for GenericIter<I, K, C>
where
    I: FusedIterator<Item = (K, C)>,
    K: Key,
    C: Chunk,
{
}
