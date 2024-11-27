use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use flat_bit_set::FlatBitSet;
use litemap::LiteMap;
use roaring::RoaringBitmap;
use std::{
    collections::{BTreeSet, HashSet},
    fmt::Arguments,
    hint::black_box,
};

type SparseBitSet = hi_sparse_bitset::BitSet<hi_sparse_bitset::config::_64bit>;

fn insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("insert");
    for n in [10, 100, 1000] {
        let mut data: Vec<_> = (0..n).collect();
        test_insert(&mut g, &data, format_args!("dense {n}"));
        data.reverse();
        test_insert(&mut g, &data, format_args!("dense rev {n}"));
        data.reverse();
        data.iter_mut().for_each(|x| *x *= 100);
        test_insert(&mut g, &data, format_args!("sparse {n}"));
        data.reverse();
        test_insert(&mut g, &data, format_args!("sparse rev {n}"));
    }
    g.finish();
}

fn test_insert(g: &mut BenchmarkGroup<WallTime>, input: &[u32], p: Arguments) {
    g.bench_with_input(BenchmarkId::new("flat_bit_set", p), &input, |b, &input| {
        b.iter_batched(
            FlatBitSet::<u32>::new,
            |mut set| {
                for &e in input {
                    set.insert(e);
                }
                set
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_with_input(BenchmarkId::new("hashset", p), &input, |b, &input| {
        b.iter_batched(
            HashSet::<u32>::new,
            |mut set| {
                for &e in input {
                    set.insert(e);
                }
                set
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_with_input(BenchmarkId::new("btreeset", p), &input, |b, &input| {
        b.iter_batched(
            BTreeSet::<u32>::new,
            |mut set| {
                for &e in input {
                    set.insert(e);
                }
                set
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_with_input(BenchmarkId::new("litemap", p), &input, |b, &input| {
        b.iter_batched(
            LiteMap::<u32, ()>::new,
            |mut set| {
                for &e in input {
                    set.insert(e, ());
                }
                set
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_with_input(BenchmarkId::new("roaring", p), &input, |b, &input| {
        b.iter_batched(
            RoaringBitmap::new,
            |mut set| {
                for &e in input {
                    set.insert(e);
                }
                set
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_with_input(
        BenchmarkId::new("hi_sparse_bitset", p),
        &input,
        |b, &input| {
            b.iter_batched(
                SparseBitSet::new,
                |mut set| {
                    for &e in input {
                        set.insert(e as usize);
                    }
                    set
                },
                BatchSize::LargeInput,
            );
        },
    );
}

fn remove(c: &mut Criterion) {
    let mut g = c.benchmark_group("remove");
    for n in [10, 100, 1000] {
        let mut data: Vec<_> = (0..n).collect();
        test_remove(&mut g, &data, format_args!("dense {n}"));
        data.reverse();
        test_remove(&mut g, &data, format_args!("dense rev {n}"));
        data.reverse();
        data.iter_mut().for_each(|x| *x *= 100);
        test_remove(&mut g, &data, format_args!("sparse {n}"));
        data.reverse();
        test_remove(&mut g, &data, format_args!("sparse rev {n}"));
    }
    g.finish();
}

fn test_remove(g: &mut BenchmarkGroup<WallTime>, input: &[u32], p: Arguments) {
    g.bench_with_input(BenchmarkId::new("flat_bit_set", p), &input, |b, &input| {
        let set: FlatBitSet<u32> = input.iter().copied().collect();
        b.iter_batched(
            || set.clone(),
            |mut set| {
                for &e in input {
                    set.remove(e);
                }
                set
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_with_input(BenchmarkId::new("hashset", p), &input, |b, &input| {
        let set: HashSet<u32> = input.iter().copied().collect();
        b.iter_batched(
            || set.clone(),
            |mut set| {
                for &e in input {
                    set.remove(&e);
                }
                set
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_with_input(BenchmarkId::new("btreeset", p), &input, |b, &input| {
        let set: BTreeSet<u32> = input.iter().copied().collect();
        b.iter_batched(
            || set.clone(),
            |mut set| {
                for &e in input {
                    set.remove(&e);
                }
                set
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_with_input(BenchmarkId::new("litemap", p), &input, |b, &input| {
        let set: LiteMap<u32, ()> = input.iter().map(|&x| (x, ())).collect();
        b.iter_batched(
            || set.clone(),
            |mut set| {
                for &e in input {
                    set.remove(&e);
                }
                set
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_with_input(BenchmarkId::new("roaring", p), &input, |b, &input| {
        let set: RoaringBitmap = input.iter().copied().collect();
        b.iter_batched(
            || set.clone(),
            |mut set| {
                for &e in input {
                    set.remove(e);
                }
                set
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_with_input(
        BenchmarkId::new("hi_sparse_bitset", p),
        &input,
        |b, &input| {
            let set: SparseBitSet = input.iter().map(|x| *x as usize).collect();
            b.iter_batched(
                || set.clone(),
                |mut set| {
                    for &e in input {
                        set.remove(e as usize);
                    }
                    set
                },
                BatchSize::LargeInput,
            );
        },
    );
}

fn get(c: &mut Criterion) {
    let mut g = c.benchmark_group("get");
    for n in [10, 100, 1000] {
        let mut data: Vec<_> = (0..n).collect();
        test_get(&mut g, &data, format_args!("dense {n}"));
        data.iter_mut().for_each(|x| *x *= 100);
        test_get(&mut g, &data, format_args!("sparse {n}"));
    }
    g.finish();
}

fn test_get(g: &mut BenchmarkGroup<WallTime>, input: &[u32], p: Arguments) {
    g.bench_with_input(BenchmarkId::new("flat_bit_set", p), &input, |b, &input| {
        let set: FlatBitSet<u32> = input.iter().copied().collect();
        b.iter(|| {
            for &e in input {
                black_box(set.contains(e));
            }
        });
    });

    g.bench_with_input(BenchmarkId::new("hashset", p), &input, |b, &input| {
        let set: HashSet<u32> = input.iter().copied().collect();
        b.iter(|| {
            for &e in input {
                black_box(set.contains(&e));
            }
        });
    });

    g.bench_with_input(BenchmarkId::new("btreeset", p), &input, |b, &input| {
        let set: BTreeSet<u32> = input.iter().copied().collect();
        b.iter(|| {
            for &e in input {
                black_box(set.contains(&e));
            }
        });
    });

    g.bench_with_input(BenchmarkId::new("litemap", p), &input, |b, &input| {
        let set: LiteMap<u32, ()> = input.iter().map(|&x| (x, ())).collect();
        b.iter(|| {
            for &e in input {
                black_box(set.get(&e).is_some());
            }
        });
    });

    g.bench_with_input(BenchmarkId::new("roaring", p), &input, |b, &input| {
        let set: RoaringBitmap = input.iter().copied().collect();
        b.iter(|| {
            for &e in input {
                black_box(set.contains(e));
            }
        });
    });

    g.bench_with_input(
        BenchmarkId::new("hi_sparse_bitset", p),
        &input,
        |b, &input| {
            let set: SparseBitSet = input.iter().map(|x| *x as usize).collect();
            b.iter(|| {
                for &e in input {
                    black_box(set.contains(e as usize));
                }
            });
        },
    );
}

fn iter(c: &mut Criterion) {
    let mut g = c.benchmark_group("iter");
    for n in [10, 100, 1000] {
        let mut data: Vec<_> = (0..n).collect();
        test_iter(&mut g, &data, format_args!("dense {n}"));
        data.iter_mut().for_each(|x| *x *= 100);
        test_iter(&mut g, &data, format_args!("sparse {n}"));
    }
    g.finish();
}

fn test_iter(g: &mut BenchmarkGroup<WallTime>, input: &[u32], p: Arguments) {
    g.bench_with_input(BenchmarkId::new("flat_bit_set", p), &input, |b, &input| {
        let set: FlatBitSet<u32> = input.iter().copied().collect();
        b.iter(|| {
            set.iter().for_each(|x| {
                black_box(x);
            });
        });
    });

    g.bench_with_input(BenchmarkId::new("hashset", p), &input, |b, &input| {
        let set: HashSet<u32> = input.iter().copied().collect();
        b.iter(|| {
            set.iter().for_each(|&x| {
                black_box(x);
            });
        });
    });

    g.bench_with_input(BenchmarkId::new("btreeset", p), &input, |b, &input| {
        let set: BTreeSet<u32> = input.iter().copied().collect();
        b.iter(|| {
            set.iter().for_each(|&x| {
                black_box(x);
            });
        });
    });

    g.bench_with_input(BenchmarkId::new("litemap", p), &input, |b, &input| {
        let set: LiteMap<u32, ()> = input.iter().map(|&x| (x, ())).collect();
        b.iter(|| {
            set.iter().for_each(|(&x, _)| {
                black_box(x);
            });
        });
    });

    g.bench_with_input(BenchmarkId::new("roaring", p), &input, |b, &input| {
        let set: RoaringBitmap = input.iter().copied().collect();
        b.iter(|| {
            set.iter().for_each(|x| {
                black_box(x);
            });
        });
    });

    g.bench_with_input(
        BenchmarkId::new("hi_sparse_bitset", p),
        &input,
        |b, &input| {
            let set: SparseBitSet = input.iter().map(|x| *x as usize).collect();
            b.iter(|| {
                set.iter().for_each(|x| {
                    black_box(x);
                });
            });
        },
    );
}

fn biop_with(c: &mut Criterion) {
    let mut g = c.benchmark_group("biop_with");
    for n in [10, 100, 1000] {
        let mut i1: Vec<_> = (0..n * 2).collect();
        let mut i2: Vec<_> = (n..n * 3).collect();
        test_biop_with(&mut g, &i1, &i2, format_args!("dense {n}"));
        i1.iter_mut().for_each(|x| *x *= 100);
        i2.iter_mut().for_each(|x| *x *= 100);
        test_biop_with(&mut g, &i1, &i2, format_args!("sparse {n}"));
    }
    g.finish();
}

fn test_biop_with(g: &mut BenchmarkGroup<WallTime>, i1: &[u32], i2: &[u32], p: Arguments) {
    g.bench_with_input(
        BenchmarkId::new("flat_bit_set", p),
        &(i1, i2),
        |b, &(i1, i2)| {
            let s1: FlatBitSet<u32> = i1.iter().copied().collect();
            let s2: FlatBitSet<u32> = i2.iter().copied().collect();
            b.iter_batched(
                || (s1.clone(), s1.clone(), s1.clone(), s1.clone()),
                |(a, b, c, d)| (a & &s2, b | &s2, c ^ &s2, d - &s2),
                BatchSize::LargeInput,
            );
        },
    );

    g.bench_with_input(BenchmarkId::new("roaring", p), &(i1, i2), |b, &(i1, i2)| {
        let s1: RoaringBitmap = i1.iter().copied().collect();
        let s2: RoaringBitmap = i2.iter().copied().collect();
        b.iter_batched(
            || (s1.clone(), s1.clone(), s1.clone(), s1.clone()),
            |(a, b, c, d)| (a & &s2, b | &s2, c ^ &s2, d - &s2),
            BatchSize::LargeInput,
        );
    });
}

fn biop_iter(c: &mut Criterion) {
    let mut g = c.benchmark_group("biop_iter");
    for n in [10, 100, 1000] {
        let mut i1: Vec<_> = (0..n * 2).collect();
        let mut i2: Vec<_> = (n..n * 3).collect();
        test_biop_iter(&mut g, &i1, &i2, format_args!("dense {n}"));
        i1.iter_mut().for_each(|x| *x *= 100);
        i2.iter_mut().for_each(|x| *x *= 100);
        test_biop_iter(&mut g, &i1, &i2, format_args!("sparse {n}"));
    }
    g.finish();
}

fn test_biop_iter(g: &mut BenchmarkGroup<WallTime>, i1: &[u32], i2: &[u32], p: Arguments) {
    g.bench_with_input(
        BenchmarkId::new("flat_bit_set", p),
        &(i1, i2),
        |b, &(i1, i2)| {
            let s1: FlatBitSet<u32> = i1.iter().copied().collect();
            let s2: FlatBitSet<u32> = i2.iter().copied().collect();
            b.iter(|| {
                for k in s1.intersection_iter(&s2) {
                    black_box(k);
                }
                for k in s1.union_iter(&s2) {
                    black_box(k);
                }
                for k in s1.symmetric_difference_iter(&s2) {
                    black_box(k);
                }
                for k in s1.difference_iter(&s2) {
                    black_box(k);
                }
            });
        },
    );

    g.bench_with_input(BenchmarkId::new("hashset", p), &(i1, i2), |b, &(i1, i2)| {
        let s1: HashSet<u32> = i1.iter().copied().collect();
        let s2: HashSet<u32> = i2.iter().copied().collect();
        b.iter(|| {
            for k in s1.intersection(&s2) {
                black_box(k);
            }
            for k in s1.union(&s2) {
                black_box(k);
            }
            for k in s1.symmetric_difference(&s2) {
                black_box(k);
            }
            for k in s1.difference(&s2) {
                black_box(k);
            }
        });
    });

    g.bench_with_input(
        BenchmarkId::new("btreeset", p),
        &(i1, i2),
        |b, &(i1, i2)| {
            let s1: BTreeSet<u32> = i1.iter().copied().collect();
            let s2: BTreeSet<u32> = i2.iter().copied().collect();
            b.iter(|| {
                for k in s1.intersection(&s2) {
                    black_box(k);
                }
                for k in s1.union(&s2) {
                    black_box(k);
                }
                for k in s1.symmetric_difference(&s2) {
                    black_box(k);
                }
                for k in s1.difference(&s2) {
                    black_box(k);
                }
            });
        },
    );
}

fn biop_new(c: &mut Criterion) {
    let mut g = c.benchmark_group("biop_new");
    for n in [10, 100, 1000] {
        let mut i1: Vec<_> = (0..n * 2).collect();
        let mut i2: Vec<_> = (n..n * 3).collect();
        test_biop_new(&mut g, &i1, &i2, format_args!("dense {n}"));
        i1.iter_mut().for_each(|x| *x *= 100);
        i2.iter_mut().for_each(|x| *x *= 100);
        test_biop_new(&mut g, &i1, &i2, format_args!("sparse {n}"));
    }
    g.finish();
}

fn test_biop_new(g: &mut BenchmarkGroup<WallTime>, i1: &[u32], i2: &[u32], p: Arguments) {
    g.bench_with_input(
        BenchmarkId::new("flat_bit_set", p),
        &(i1, i2),
        |b, &(i1, i2)| {
            let s1: FlatBitSet<u32> = i1.iter().copied().collect();
            let s2: FlatBitSet<u32> = i2.iter().copied().collect();
            b.iter_with_large_drop(|| (&s1 & &s2, &s1 | &s2, &s1 ^ &s2, &s1 - &s2));
        },
    );

    g.bench_with_input(BenchmarkId::new("roaring", p), &(i1, i2), |b, &(i1, i2)| {
        let s1: RoaringBitmap = i1.iter().copied().collect();
        let s2: RoaringBitmap = i2.iter().copied().collect();
        b.iter_with_large_drop(|| (&s1 & &s2, &s1 | &s2, &s1 ^ &s2, &s1 - &s2));
    });
}

criterion_group!(benches, insert, remove, get, iter, biop_with, biop_iter, biop_new);
criterion_main!(benches);
