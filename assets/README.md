# Local assets

The warehouse demo does **not** vendor any large USD files. It streams the
NVIDIA SimReady props and Isaac Sim robot assets from the Isaac Sim asset root
configured in the local Isaac Sim installation (the standard 5.1 asset root is
used by default).

`assets/lekiwi/` is kept as an optional local workspace for the older LeKiwi
import helpers. Nothing in the warehouse demo depends on files in this folder.

`assets/warehouse_start.usda` is the minimal template stage opened by the
animated People crew through the Isaac Sim USD context. The demo builds the
real simulation stage first and the People layer then attaches the workers;
the template keeps the default `World` prim with `metersPerUnit = 1.0` and
`upAxis = "Z"`.
