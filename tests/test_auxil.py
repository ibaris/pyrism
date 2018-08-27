import numpy as np
import pytest

from pyrism.core import (ReflectanceResult, EmissivityResult, SailResult)


class TestResultClass:
    def test_reflectance(self):
        test = ReflectanceResult(a=1, b=2)
        assert test.a == 1
        assert test.b == 2

    def test_emissivity(self):
        test = EmissivityResult(a=1, b=2)
        assert test.a == 1
        assert test.b == 2

    def test_sail(self):
        test = SailResult(a=1, b=2)
        assert test.a == 1
        assert test.b == 2
