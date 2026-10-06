from types import SimpleNamespace

import pytest

from b_asic.code_printer import VhdlPrinter
from b_asic.code_printer.vhdl.util import signed_type
from b_asic.data_type import DataType, NumRepresentation


def test_signed_type():
    assert signed_type(4) == "signed(3 downto 0)"


class TestVhdlPrinterFloatingPoint:
    """Unit tests for floating-point VHDL code generation."""

    @pytest.fixture
    def fp_printer(self):
        """Create a VhdlPrinter with floating-point data type (single precision)."""
        dt = DataType(wl=(8, 23), num_repr=NumRepresentation.FLOATING_POINT)
        return VhdlPrinter(dt)

    def test_print_SquareRoot_non_amd_returns_default(self, fp_printer):
        """SquareRoot with non-AMD backend should return default (empty code)."""
        fp_printer._fp_backend = ""

        # Call with None since the method doesn't use pe parameter for SquareRoot
        wls, (declarations, code) = fp_printer.print_SquareRoot_floating_point_real(
            None
        )

        assert declarations == ""
        assert code == ""
        assert wls == [fp_printer._dt.wl]

    def test_print_SquareRoot_amd_backend(self, fp_printer):
        """SquareRoot with AMD backend should use _amd_fp_backend."""
        fp_printer._fp_backend = "amd"

        wls, (declarations, code) = fp_printer.print_SquareRoot_floating_point_real(
            None
        )

        # Should have component declarations and instantiation
        assert "fp_sqrt" in declarations
        assert "u_fp_sqrt" in code
        assert "port map" in code
        assert wls == [fp_printer._dt.wl]

    def test_print_Absolute_clears_sign_bit(self, fp_printer):
        """Absolute should clear the sign bit (force to '0')."""
        wls, (declarations, code) = fp_printer.print_Absolute_floating_point_real(None)

        # Check that sign bit is cleared
        assert "res_arith_0" in declarations
        assert "'0' & op_0" in code
        assert f"op_0({fp_printer.bits - 2} downto 0)" in code
        assert wls == [fp_printer._dt.wl]

    def test_print_Sign_constructs_plus_minus_one(self, fp_printer):
        """Sign should output ±1.0 with correct exponent and zero mantissa."""
        wls, (declarations, code) = fp_printer.print_Sign_floating_point_real(None)

        # Check that all components are present
        assert "res_arith_0" in declarations
        assert f"op_0({fp_printer.bits - 1})" in code  # sign bit extraction
        assert "'0'" in code  # exponent MSB
        assert f"({fp_printer.exp_bits - 2} downto 0 => '1')" in code  # exponent rest
        assert f"({fp_printer.man_bits - 1} downto 0 => '0')" in code  # mantissa zeros
        assert wls == [fp_printer._dt.wl]

    def test_print_Sign_exponent_pattern_single_precision(self, fp_printer):
        """Sign exponent should be bias (all 1s except MSB) for both +1.0 and -1.0."""
        # For single precision: exp_bits=8, so bias=127=01111111 binary
        # MSB is 0, followed by 7 ones
        assert fp_printer.exp_bits == 8
        assert fp_printer.man_bits == 23

        wls, (declarations, code) = fp_printer.print_Sign_floating_point_real(None)

        # For 8-bit exponent: exp_bits - 2 = 6, so (6 downto 0 => '1') gives 7 ones
        assert "(6 downto 0 => '1')" in code
        # For 23-bit mantissa: (22 downto 0 => '0')
        assert "(22 downto 0 => '0')" in code

    def test_print_Absolute_constant_value(self, fp_printer):
        """Test Absolute output code structure."""
        wls, (declarations, code) = fp_printer.print_Absolute_floating_point_real(None)

        # Generate expected VHDL
        expected_code = f"res_arith_0 <= '0' & op_0({fp_printer.bits - 2} downto 0);"
        assert expected_code in code

    def test_print_Sign_constant_value(self, fp_printer):
        """Test Sign output code structure."""
        wls, (declarations, code) = fp_printer.print_Sign_floating_point_real(None)

        # Generate expected VHDL pattern
        # res_arith_0 <= op_0(31) & '0' & (6 downto 0 => '1') & (22 downto 0 => '0');
        expected_pattern = f"res_arith_0 <= op_0({fp_printer.bits - 1}) & '0' & ({fp_printer.exp_bits - 2} downto 0 => '1') & ({fp_printer.man_bits - 1} downto 0 => '0');"
        assert expected_pattern in code

    def test_print_Constant_static(self, fp_printer):
        """Static floating-point constant should generate a direct assignment."""
        pe = SimpleNamespace(
            processes=[
                SimpleNamespace(start_time=0, operation=SimpleNamespace(value=1.5))
            ],
            schedule_time=1,
        )

        wls, (declarations, code) = fp_printer.print_Constant_floating_point_real(pe)

        assert "res_arith_0" in declarations
        assert "res_arith_0 <=" in code
        assert 'b"00111111110000000000000000000000"' in code
        assert "with schedule_cnt select" not in code
        assert wls == [fp_printer._dt.wl]

    def test_print_Constant_dynamic(self, fp_printer):
        """Time-varying floating-point constants should generate schedule-based muxing."""
        pe = SimpleNamespace(
            processes=[
                SimpleNamespace(start_time=0, operation=SimpleNamespace(value=1.0)),
                SimpleNamespace(start_time=2, operation=SimpleNamespace(value=-2.0)),
            ],
            schedule_time=4,
        )

        wls, (declarations, code) = fp_printer.print_Constant_floating_point_real(pe)

        assert "with schedule_cnt select" in code
        assert 'when "00"' in code
        assert 'when "10"' in code
        assert 'b"00111111100000000000000000000000"' in code
        assert 'b"11000000000000000000000000000000"' in code
        assert "when others;" in code
        assert wls == [fp_printer._dt.wl]
