//! 二元逻辑运算
//!
//! [`FlatBitSet`] 和 [`Thunk`] 提供了 `&`、`|`、`^`、`-` 运算以及一些同类的方法，请优先用之。

use crate::*;
use core::{
    cmp::Ordering,
    iter::{FusedIterator, Peekable},
    marker::PhantomData,
};

/// 二元逻辑运算
///
/// 运算针对 [`Chunk`] 而非布尔值。
pub trait BiOp<C: Chunk> {
    /// 只有第一个
    #[inline]
    fn first(c1: C) -> C {
        Self::both(c1, C::ZERO)
    }

    /// 只有第二个
    #[inline]
    fn second(c2: C) -> C {
        Self::both(C::ZERO, c2)
    }

    /// 俩都有
    fn both(c1: C, c2: C) -> C;
}

/// 与
#[derive(Clone, Copy)]
pub enum AndOp {}

/// 或
#[derive(Clone, Copy)]
pub enum OrOp {}

/// 异或
#[derive(Clone, Copy)]
pub enum XorOp {}

/// 差
#[derive(Clone, Copy)]
pub enum DiffOp {}

impl<C: Chunk> BiOp<C> for AndOp {
    #[inline]
    fn both(c1: C, c2: C) -> C {
        c1 & c2
    }
}

impl<C: Chunk> BiOp<C> for OrOp {
    #[inline]
    fn both(c1: C, c2: C) -> C {
        c1 | c2
    }
}

impl<C: Chunk> BiOp<C> for XorOp {
    #[inline]
    fn both(c1: C, c2: C) -> C {
        c1 ^ c2
    }
}

impl<C: Chunk> BiOp<C> for DiffOp {
    #[inline]
    fn both(c1: C, c2: C) -> C {
        c1 & !c2
    }
}

/// 二元逻辑运算迭代器
///
/// 请使用 [`Thunk::into_iter`]，或者 [`FlatBitSet::intersection_iter`] 等一系列方法。
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct BiOpIter<O, I1, I2>
where
    I1: Iterator,
    I1::Item: Clone,
    I2: Iterator,
    I2::Item: Clone,
{
    i1: Peekable<I1>,
    i2: Peekable<I2>,
    _marker: PhantomData<O>,
}

impl<O, I1, I2> BiOpIter<O, I1, I2>
where
    I1: Iterator,
    I1::Item: Clone,
    I2: Iterator,
    I2::Item: Clone,
{
    pub(crate) fn new(i1: I1, i2: I2) -> Self {
        Self {
            i1: i1.peekable(),
            i2: i2.peekable(),
            _marker: PhantomData,
        }
    }
}

impl<K, C, I1, I2, O> Iterator for BiOpIter<O, I1, I2>
where
    K: Key,
    C: Chunk,
    I1: Iterator<Item = (K, C)>,
    I2: Iterator<Item = (K, C)>,
    O: BiOp<C>,
{
    type Item = (K, C);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            debug_assert!(self.i1.peek().map_or(true, |&(_, c)| c != C::ZERO));
            debug_assert!(self.i2.peek().map_or(true, |&(_, c)| c != C::ZERO));
            let (k, c) = match (self.i1.peek(), self.i2.peek()) {
                (None, None) => None?,
                (None, Some(&(k2, c2))) => {
                    self.i2.next();
                    (k2, O::second(c2))
                }
                (Some(&(k1, c1)), None) => {
                    self.i1.next();
                    (k1, O::first(c1))
                }
                (Some(&(k1, c1)), Some(&(k2, c2))) => match k1.cmp(&k2) {
                    Ordering::Less => {
                        self.i1.next();
                        (k1, O::first(c1))
                    }
                    Ordering::Equal => {
                        self.i1.next();
                        self.i2.next();
                        (k1, O::both(c1, c2))
                    }
                    Ordering::Greater => {
                        self.i2.next();
                        (k2, O::second(c2))
                    }
                },
            };
            if c != C::ZERO {
                break Some((k, c));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let hi = Option::zip(self.i1.size_hint().1, self.i2.size_hint().1);
        (0, hi.and_then(|(x, y)| x.checked_add(y)))
    }
}

impl<K, C, I1, I2, O> FusedIterator for BiOpIter<O, I1, I2>
where
    K: Key,
    C: Chunk,
    I1: Iterator<Item = (K, C)>,
    I2: Iterator<Item = (K, C)>,
    O: BiOp<C>,
{
}
