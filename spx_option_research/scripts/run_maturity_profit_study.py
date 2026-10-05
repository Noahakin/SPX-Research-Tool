"""Run the complete maturity grid, independent ledger audit and readable report."""
from maturity_grid_data import prepare_paths
from run_maturity_profit_grid import run
from validate_maturity_grid import validate
from present_maturity_grid import present
from historical_maturity_grid import historical


if __name__ == '__main__':
    prepared = prepare_paths()
    run(prepared)
    historical(prepared)
    validate(prepared)
    present()
