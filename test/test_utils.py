#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.utils import get_inheritors, flatten, convert_to_nested_indices


def test_get_inheritors():
    class A(object):
        pass

    class B(A):
        pass

    class C(B):
        pass

    class D(A):
        pass

    class E(object):
        pass

    assert get_inheritors(A) == {B, C, D}


def test_flatten():
    assert list(flatten([0, (1, 2, 3), 4, [5, (6, 7), 8]])) == list(range(9))


def test_convert_to_nested_indices():
    assert list(convert_to_nested_indices(
        ['a', ('b', 'c', 'a'), 'b', ['c', ('c', 'c'), 'b']])) == \
        [0, [1, 2, 3], 4, [5, [6, 7], 8]]
