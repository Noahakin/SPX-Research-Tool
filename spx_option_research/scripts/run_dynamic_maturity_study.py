"""Run candidate preparation, dynamic policy grid, audits and report."""
from maturity_grid_data import prepare_paths
from dynamic_maturity_data import build
from run_dynamic_maturity_search import run
from finish_dynamic_maturity_search import finish
from present_dynamic_maturity_search import present
from dynamic_maturity_fixed_controls import compare


if __name__=='__main__':
    data=build(prepare_paths())
    state=run(data)
    finish(data,state)
    compare(data)
    present()
