# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.pump_component import Pump


class Compressor(Pump):
    """

    """

    @classmethod
    def table_name(cls):
        return "compressor"

    @classmethod
    def create_pit_branch_entries(cls, net, compressor_pit, node_name):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param compressor_pit: a part of the pit that includes only those columns relevant for
                               compressors
        :type compressor_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        compressor_pit = super(Pump, cls).create_pit_branch_entries(net, compressor_pit, node_name)

        compressor_pit[:, net['_idx_branch']['D']] = 0.1
        compressor_pit[:, net['_idx_branch']['AREA']] = compressor_pit[:, net['_idx_branch']['D']] ** 2 * np.pi / 4
        compressor_pit[:, net['_idx_branch']['LOSS_COEFFICIENT']] = 0
        compressor_pit[:, net['_idx_branch']['PRESSURE_RATIO']] = net[cls.table_name()].pressure_ratio.values

    @classmethod
    def calculate_pressure_lift(cls, net, compressor_pit, node_pit):
        """ absolute inlet pressure multiplied by the compressor's boost ratio.

        If the flow is reversed, the pressure lift is set to 0.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param compressor_pit:
        :type compressor_pit:
        :param node_pit:
        :type node_pit:
        """
        pressure_ratio = compressor_pit[:, net['_idx_branch']['PRESSURE_RATIO']]

        from_nodes = compressor_pit[:, net['_idx_branch']['FROM_NODE']].astype(np.int32)
        p_from = node_pit[from_nodes, net['_idx_node']['PAMB']] + node_pit[from_nodes, net['_idx_node']['PINIT']]
        p_to_calc = p_from * pressure_ratio
        pl_abs = p_to_calc - p_from

        v_mps = compressor_pit[:, net['_idx_branch']['VINIT']]
        pl_abs *= (v_mps >= 0)  # force pressure lift = 0 for reverse flow

        compressor_pit[:, net['_idx_branch']['PL']] = pl_abs

    @classmethod
    def get_component_input(cls):
        """

        Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("pressure_ratio", "f8"),
                ("in_service", 'bool')]