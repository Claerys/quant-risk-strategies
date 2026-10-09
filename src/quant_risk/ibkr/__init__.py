"""Interactive Brokers connection: download futures history and route paper orders.

Adapted from the BSQF ibkr_base project (used with its authors' permission). Importing this
package never imports ``ibapi``; only the modules that open a connection do, so the rest of
the project and its tests run without it. Install it with ``pip install -e ".[ibkr]"``.
"""
