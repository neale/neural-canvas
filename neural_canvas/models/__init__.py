from .inr_maps_2d import (
    INRRandomGraph,
    INRLSimpleLinearMap,
    INRConvMap,
    INRLinearMap
)

from .inr_maps_3d import (
    INRRandomGraph3D,
    INRLinearMap3D,
    INRConvMap3D,
)

from .inrf import (
    INRF2D,
    INRF3D
)

from .ops import (
    Gaussian,
    SinLayer,
    CosLayer,
    ScaleAct,
    AddAct,
    STEFunction,
    StraightThroughEstimator
)

from .torchgraph import (
    get_graph_info,
    build_random_graph,
    plot_graph,
    randact,
    ScaleOp,
    AddOp,
    LinearActOp,
    ConvActOp,
    RandNodeOP,
    RandOp,
    TorchGraph,
)

from .weight_inits import (
    init_weights_normal,
    init_weights_uniform,
    init_weights_dip,
    init_weights_siren
)

from .discriminator_vqgan import (
    ActNorm,
    NLayerDiscriminator
)