# Copyright (c) Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from enum import Enum
import glob
import json
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from statistics import mean, stdev
from glob import glob
from itertools import cycle
from re import findall, search, split
from copy import deepcopy
from collections import defaultdict

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# A simple python script to plot measurements results. This script requires
# the following dependencies: `pip install matplotlib`.

# -- Mysticeti data parser --


class MysticetiData:
    def __init__(self, data_directory, nodes, faults, transaction_size, workload):
        self.data_directory = data_directory
        self.nodes = nodes
        self.faults = faults
        self.transaction_size = transaction_size
        self.workload = workload

    def _load_measurement_data(self, filename):
        measurements = []
        files = glob(os.path.join(self.data_directory, filename))
        for file in files:
            with open(file, "r") as f:
                try:
                    measurements += [json.loads(f.read())]
                except json.JSONDecodeError as e:
                    raise PlotError(f"Failed to load file {file}: {e}")

        return measurements

    @staticmethod
    def _file_format(transaction_size, faults, nodes, load):
        return f"measurements-{transaction_size}-{faults}-{nodes}-{load}.json"

    @classmethod
    def _ramp_up(cls, scraper, ramp_up_threshold=120):
        ramp_up_duration, ramp_up_count = 0, 0
        ramp_up_sum, ramp_up_square_sum = 0, 0
        for data in scraper:
            duration = float(data["timestamp"]["secs"])
            if duration > ramp_up_threshold:
                ramp_up_duration = duration
                ramp_up_count = float(data["count"])
                ramp_up_sum = float(data["sum"]["secs"])
                ramp_up_square_sum = float(data["squared_sum"]["secs"])
                break
        return ramp_up_duration, ramp_up_count, ramp_up_sum, ramp_up_square_sum

    @classmethod
    def _aggregate_tps(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        max_duration = 0
        for data in measurement["data"][workload].values():
            ramp_up_duration, _, _, _ = cls._ramp_up(data)
            duration = float(data[-1]["timestamp"]["secs"]) - ramp_up_duration
            max_duration = max(duration, max_duration)

        tps = []
        for data in measurement["data"][workload].values():
            _, ramp_up_count, _, _ = cls._ramp_up(data)
            count = float(data[-1]["count"]) - ramp_up_count
            tps += [(count / max_duration) if max_duration != 0 else 0]
        return max(tps)

    @classmethod
    def _aggregate_average_latency(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        latency = []
        for data in measurement["data"][workload].values():
            _, ramp_up_count, ramp_up_sum, _ = cls._ramp_up(data)
            last = data[-1]
            count = float(last["count"]) - ramp_up_count
            total = float(last["sum"]["secs"]) - ramp_up_sum
            latency += [total / count if count != 0 else 0]
        return sum(latency) / len(latency) if latency else 0

    @classmethod
    def _aggregate_stdev_latency(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        stdev = []
        for data in measurement["data"][workload].values():
            _, ramp_up_count, ramp_up_sum, ramp_up_square_sum = cls._ramp_up(data)
            last = data[-1]
            count = float(last["count"]) - ramp_up_count
            if count == 0:
                stdev += [0]
            else:
                latency_sum = float(last["sum"]["secs"]) - ramp_up_sum
                latency_square_sum = (
                    float(last["squared_sum"]["secs"]) - ramp_up_square_sum
                )

                first_term = latency_square_sum / count
                second_term = (latency_sum / count) ** 2
                stdev += [math.sqrt(first_term - second_term)]
        return max(stdev)

    def _to_plot_data_id(self, measurement, max_latency=None):
        nodes = measurement["parameters"]["nodes"]

        if "Permanent" in measurement["parameters"]["faults"]:
            faults = measurement["parameters"]["faults"]["Permanent"]["faults"]
        else:
            faults = 0

        if self.workload.casefold() == "owned".casefold():
            system = f"Mysticeti-FPC"
        else:
            system = "Mysticeti"

        self.max_latency = max_latency

        return PlotDataId(system, nodes, faults, max_latency)

    def for_latency_throughput_plot(self):
        filename = self._file_format(
            self.transaction_size, self.faults, self.nodes, "*"
        )
        measurements = self._load_measurement_data(filename)
        measurements.sort(key=lambda x: x["parameters"]["load"])

        x_values, y_values, e_values = [], [], []
        for measurement in measurements:
            x_values += [self._aggregate_tps(measurement, self.workload)]
            y_values += [self._aggregate_average_latency(measurement, self.workload)]
            e_values += [self._aggregate_stdev_latency(measurement, self.workload)]

        if x_values:
            id = self._to_plot_data_id(measurements[0])
            return PlotData(id, x_values, y_values, e_values)

    def for_scalability_plot(self, max_latency):
        data = []
        for n in self.nodes:
            filename = self._file_format(self.transaction_size, self.faults, n, "*")
            measurements = self._load_measurement_data(filename)
            measurements = [
                x
                for x in measurements
                if self._aggregate_average_latency(x, self.workload) <= self.max_latency
            ]
            if measurements:
                data += [
                    max(
                        measurements,
                        key=lambda x: self._aggregate_tps(x, self.workload),
                    )
                ]

        x_values, y_values, e_values = [], [], []
        for measurement in data:
            x_values += [measurement["parameters"]["nodes"]]
            y_values += [self._aggregate_tps(measurement, self.workload)]
            e_values += [0]

        if x_values:
            id = self._to_plot_data_id(data[0], max_latency)
            return PlotData(id, x_values, y_values, e_values)


# -- New Mysticeti data parser --


class MysticetiDataNew:
    def __init__(
        self, data_directory, nodes, faults, transaction_size, workload, parameters=""
    ):
        self.data_directory = data_directory
        self.nodes = nodes
        self.faults = faults
        self.transaction_size = transaction_size
        self.workload = workload
        self.parameters = parameters

    def _load_measurement_data(self, filename):
        measurements = []
        files = glob(os.path.join(self.data_directory, filename))
        for file in files:
            with open(file, "r") as f:
                try:
                    measurements += [json.loads(f.read())]
                except json.JSONDecodeError as e:
                    raise PlotError(f"Failed to load file {file}: {e}")

        return measurements

    @staticmethod
    def _file_format(transaction_size, faults, nodes, load, parameters=""):
        return f"measurements-c-{transaction_size}-{faults}-{nodes}-{load}{parameters}.json"

    @classmethod
    def _ramp_up(cls, scraper, ramp_up_threshold=None):
        if ramp_up_threshold is None:
            ramp_up_threshold = float(scraper[-1]["timestamp"]["secs"]) - 120

        ramp_up_duration, ramp_up_count = 0, 0
        ramp_up_sum, ramp_up_square_sum = 0, 0
        for data in scraper:
            duration = float(data["timestamp"]["secs"])
            if duration > ramp_up_threshold:
                ramp_up_duration = duration
                ramp_up_count = float(data["count"])
                ramp_up_sum = float(data["sum"]["secs"])
                ramp_up_square_sum = float(data["squared_sum"])
                break
        return ramp_up_duration, ramp_up_count, ramp_up_sum, ramp_up_square_sum

    @classmethod
    def _aggregate_tps(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        max_duration = 0
        for data in measurement["data"][workload].values():
            ramp_up_duration, _, _, _ = cls._ramp_up(data)
            duration = float(data[-1]["timestamp"]["secs"]) - ramp_up_duration
            max_duration = max(duration, max_duration)

        tps = []
        for data in measurement["data"][workload].values():
            _, ramp_up_count, _, _ = cls._ramp_up(data)
            count = float(data[-1]["count"]) - ramp_up_count
            tps += [(count / max_duration) if max_duration != 0 else 0]
        return max(tps)

    @classmethod
    def _aggregate_average_latency(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        latency = []
        for data in measurement["data"][workload].values():
            _, ramp_up_count, ramp_up_sum, _ = cls._ramp_up(data)
            last = data[-1]
            count = float(last["count"]) - ramp_up_count
            total = float(last["sum"]["secs"]) - ramp_up_sum
            latency += [total / count if count != 0 else 0]
        return sum(latency) / len(latency) if latency else 0

    @classmethod
    def _aggregate_stdev_latency(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        stdev = []
        for data in measurement["data"][workload].values():
            _, ramp_up_count, ramp_up_sum, ramp_up_square_sum = cls._ramp_up(data)
            last = data[-1]
            count = float(last["count"]) - ramp_up_count
            if count == 0:
                stdev += [0]
            else:
                latency_sum = float(last["sum"]["secs"]) - ramp_up_sum
                latency_square_sum = float(last["squared_sum"]) - ramp_up_square_sum

                first_term = latency_square_sum / count
                second_term = (latency_sum / count) ** 2
                stdev += [math.sqrt(first_term - second_term)]
        return max(stdev)

    @classmethod
    def _latency_time_series(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        latencies = []  # list of time series
        for data in measurement["data"][workload].values():
            _, ramp_up_count, ramp_up_sum, _ = cls._ramp_up(data)
            series = []
            for d in data:
                count = float(d["count"]) - ramp_up_count
                total = float(d["sum"]["secs"]) - ramp_up_sum
                latency = total / count if count != 0 else 0
                timestamp = float(d["timestamp"]["secs"])
                series.append((timestamp, latency))
            latencies += [series]
        return latencies

    def _to_plot_data_id(self, measurement, max_latency=None):
        nodes = measurement["parameters"]["nodes"]
        faults = measurement["parameters"]["settings"]["faults"]

        if "Permanent" in faults:
            faults = faults["Permanent"]["faults"]
        else:
            faults = 0

        if self.workload.casefold() == "owned".casefold():
            system = f"Mysticeti-FPC"
        else:
            system = "Mysticeti"

        self.max_latency = max_latency

        return PlotDataId(system, nodes, faults, max_latency)

    def for_latency_throughput_plot(self):
        filename = self._file_format(
            self.transaction_size, self.faults, self.nodes, "*", self.parameters
        )
        measurements = self._load_measurement_data(filename)
        measurements.sort(key=lambda x: x["parameters"]["load"])

        x_values, y_values, e_values = [], [], []
        for measurement in measurements:
            x_values += [self._aggregate_tps(measurement, self.workload)]
            y_values += [self._aggregate_average_latency(measurement, self.workload)]
            e_values += [self._aggregate_stdev_latency(measurement, self.workload)]

        if x_values:
            id = self._to_plot_data_id(measurements[0])
            return PlotData(id, x_values, y_values, e_values)

    def for_histogram(self, label, load):
        filename = self._file_format(
            self.transaction_size,
            self.faults,
            self.nodes,
            load,
            self.parameters,
        )

        measurements = self._load_measurement_data(filename)
        if not measurements:
            return None

        measurement = measurements[0]

        x_values = [label]
        y_values = [self._aggregate_average_latency(measurement, self.workload)]
        e_values = []

        if x_values:
            id = self._to_plot_data_id(measurement)
            return PlotData(id, x_values, y_values, e_values)


# -- Obelia data parser --


class ObeliaData:
    def __init__(
        self,
        data_directory,
        faults,
        nodes,
        aux,
        transaction_size,
        workload,
        parameters="",
    ):
        self.data_directory = data_directory
        self.nodes = nodes
        self.aux = aux
        self.faults = faults
        self.transaction_size = transaction_size
        self.workload = workload
        self.parameters = parameters

    def _load_measurement_data(self, filename):
        measurements = []
        files = glob(os.path.join(self.data_directory, filename))
        for file in files:
            with open(file, "r") as f:
                try:
                    measurements += [json.loads(f.read())]
                except json.JSONDecodeError as e:
                    raise PlotError(f"Failed to load file {file}: {e}")

        return measurements

    @staticmethod
    def _file_format(transaction_size, faults, nodes, aux, load, parameters=""):
        return f"measurements-c-{transaction_size}-{faults}-{nodes}-{aux}-{load}{parameters}.json"

    @classmethod
    def _ramp_up(cls, scraper, ramp_up_threshold=None):
        if ramp_up_threshold is None:
            ramp_up_threshold = float(scraper[-1]["timestamp"]["secs"]) - 30

        ramp_up_duration, ramp_up_count = 0, 0
        ramp_up_sum, ramp_up_square_sum = 0, 0
        for data in scraper:
            duration = float(data["timestamp"]["secs"])
            if duration > ramp_up_threshold:
                ramp_up_duration = duration
                ramp_up_count = float(data["count"])
                ramp_up_sum = float(data["sum"]["secs"])
                ramp_up_square_sum = float(data["squared_sum"])
                break
        return ramp_up_duration, ramp_up_count, ramp_up_sum, ramp_up_square_sum

    @classmethod
    def _aggregate_tps(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        max_duration = 0
        for data in measurement["data"][workload].values():
            ramp_up_duration, _, _, _ = cls._ramp_up(data)
            duration = float(data[-1]["timestamp"]["secs"]) - ramp_up_duration
            max_duration = max(duration, max_duration)

        tps = []
        for data in measurement["data"][workload].values():
            _, ramp_up_count, _, _ = cls._ramp_up(data)
            count = float(data[-1]["count"]) - ramp_up_count
            tps += [(count / max_duration) if max_duration != 0 else 0]
        return max(tps)

    @classmethod
    def _aggregate_average_latency(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        latency = []
        for data in measurement["data"][workload].values():
            _, ramp_up_count, ramp_up_sum, _ = cls._ramp_up(data)
            last = data[-1]
            count = float(last["count"]) - ramp_up_count
            total = float(last["sum"]["secs"]) - ramp_up_sum
            latency += [total / count if count != 0 else 0]
        return sum(latency) / len(latency) if latency else 0

    @classmethod
    def _aggregate_stdev_latency(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        stdev = []
        for data in measurement["data"][workload].values():
            _, ramp_up_count, ramp_up_sum, ramp_up_square_sum = cls._ramp_up(data)
            last = data[-1]
            count = float(last["count"]) - ramp_up_count
            if count == 0:
                stdev += [0]
            else:
                latency_sum = float(last["sum"]["secs"]) - ramp_up_sum
                latency_square_sum = float(last["squared_sum"]) - ramp_up_square_sum

                first_term = latency_square_sum / count
                second_term = (latency_sum / count) ** 2
                stdev += [math.sqrt(first_term - second_term)]
        return max(stdev)

    @classmethod
    def _latency_time_series(cls, measurement, workload):
        if workload not in measurement["data"]:
            return 0

        latencies = []  # list of time series
        for data in measurement["data"][workload].values():
            _, ramp_up_count, ramp_up_sum, _ = cls._ramp_up(data)
            series = []
            for d in data:
                count = float(d["count"]) - ramp_up_count
                total = float(d["sum"]["secs"]) - ramp_up_sum
                latency = total / count if count != 0 else 0
                timestamp = float(d["timestamp"]["secs"])
                series.append((timestamp, latency))
            latencies += [series]
        return latencies

    def _to_plot_data_id(self, measurement, max_latency=None):
        # Ugly hack
        nodes = measurement["parameters"]["nodes"] - self.aux + self.faults
        faults = measurement["parameters"]["settings"]["faults"]

        if "Permanent" in faults:
            faults = faults["Permanent"]["faults"]
        else:
            faults = 0

        system = "Obelia"
        self.max_latency = max_latency

        id = PlotDataId(system, nodes, faults, self.aux, max_latency)
        id.faults = self.faults
        return id

    def for_latency_throughput_plot(self):
        filename = self._file_format(
            self.transaction_size,
            self.faults,
            self.nodes,
            self.aux,
            "*",
            self.parameters,
        )
        measurements = self._load_measurement_data(filename)
        measurements.sort(key=lambda x: x["parameters"]["load"])

        x_values, y_values, e_values = [], [], []
        for measurement in measurements:
            x_values += [self._aggregate_tps(measurement, self.workload)]
            y_values += [self._aggregate_average_latency(measurement, self.workload)]
            e_values += [self._aggregate_stdev_latency(measurement, self.workload)]

        if x_values:
            id = self._to_plot_data_id(measurements[0])
            return PlotData(id, x_values, y_values, e_values)

    def for_histogram(self, label, load):
        filename = self._file_format(
            self.transaction_size,
            self.faults,
            self.nodes,
            self.aux,
            load,
            self.parameters,
        )

        measurements = self._load_measurement_data(filename)
        if not measurements:
            return None

        measurement = measurements[0]

        x_values = [label]
        y_values = [self._aggregate_average_latency(measurement, self.workload)]
        e_values = []

        if x_values:
            id = self._to_plot_data_id(measurement)
            return PlotData(id, x_values, y_values, e_values)


# -- Plotter --


@tick.FuncFormatter
def default_major_formatter(x, pos):
    if pos is not None:
        return f"{x/1000:.0f}k" if x >= 10_000 else f"{x:,.0f}"


@tick.FuncFormatter
def sec_major_formatter(x, pos):
    if pos is not None:
        return f"{x:,.0f}" if x >= 10 else f"{x:,.1f}"


class PlotError(Exception):
    pass


class PlotType(Enum):
    L_GRAPH = 1
    SCALABILITY = 2
    TIME_SERIES = 3
    HIST = 4


class PlotDataId:
    def __init__(self, system, nodes, faults, aux=0, max_latency=None):
        self.system = system
        self.nodes = nodes
        self.faults = faults
        self.aux = aux
        self.max_latency = max_latency


class PlotData:
    def __init__(self, data_id, x_values, y_values, e_values):
        self.id = data_id
        self.x_values = x_values
        self.y_values = y_values
        self.e_values = e_values


class Plotter:
    def __init__(
        self,
        plot_name,
        plot_type,
        y_max=60,
        legend_columns=2,
        legend_location="upper center",
        legend_anchor=(0.5, 1),
        x_max=None,
        figure_size=(6.4, 2.4),
        yscale=None,
        frameon=True,
    ):
        self.plot_name = plot_name
        self.plot_type = plot_type
        self.y_max = y_max
        self.x_max = x_max
        self.legend_columns = legend_columns
        self.legend_location = legend_location
        self.legend_anchor = legend_anchor
        self.yscale = yscale
        self.frameon = frameon

        self.colors = cycle(["tab:green", "tab:blue", "tab:orange", "tab:red"])
        self.markers = cycle(["o", "v", "s", "d"])
        self.linestyles = cycle(["solid", "dotted"])
        self.patterns = cycle(["", "//"])

        plt.figure(figsize=figure_size)

    def new_system(self):
        self.color = next(self.colors)
        self.marker = next(self.markers)

    def _make_plot_directory(self):
        plot_directory = "plots"
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        return plot_directory

    def _legend_entry(self, plot_type, id):
        if plot_type == PlotType.L_GRAPH or plot_type == PlotType.HIST:
            if id.system == "Obelia":
                f = "" if id.faults == 0 else f" ({id.faults} faulty)"
                # a = "" if id.aux == 0 else f",\n{id.aux} aux"
                # l = f"{id.nodes} core{a}{f}"
                # return f"{id.system} - {l}"
                return f"O-{id.nodes}-{id.aux}{f}"
            elif id.system == "Mysticeti":
                # f = "" if id.faults == 0 else f" ({id.faults} faulty)"
                # l = f"{id.nodes} nodes{f}"
                # return f"{id.system} - {l}"
                return f"M-{id.nodes}"

        return None

    def _axes_labels(self, plot_type):
        if plot_type == PlotType.L_GRAPH:
            return ("Throughput (tx/s)", "Latency (s)")
        elif plot_type == PlotType.SCALABILITY:
            return ("Committee size", "Throughput (tx/s)")
        elif plot_type == PlotType.TIME_SERIES:
            return ("Time (s)", "Latency (s)")
        elif plot_type == PlotType.HIST:
            return ("Systems at 50k tx/s", "Latency (s)")
        else:
            assert False

    def plot(self, data):
        if data is None:
            return

        e_values = [x if x < self.y_max else 0 for x in data.e_values]

        plt.errorbar(
            data.x_values,
            data.y_values,
            yerr=e_values,
            label=self._legend_entry(self.plot_type, data.id),
            linestyle=next(self.linestyles),
            color=self.color,
            marker=self.marker,
            capsize=3,
            linewidth=4,
            markersize=10,
        )

    def bar(self, data):
        if data is None:
            return

        plt.bar(
            data.x_values,
            data.y_values,
            color=self.color,
            hatch=next(self.patterns),
            label=self._legend_entry(self.plot_type, data.id),
            capsize=3,
        )
        # plt.xticks([""] * len(data.x_values))
        # plt.xticks([])
        # plt.xticks(rotation=45, ha="right")

    def finalize(self):
        x_label, y_label = self._axes_labels(self.plot_type)

        legend_properties = {"weight": "bold"}
        plt.legend(
            loc=self.legend_location,
            bbox_to_anchor=self.legend_anchor,
            ncol=self.legend_columns,
            frameon=self.frameon,
            prop=legend_properties,
        )

        if not (self.plot_type == PlotType.HIST):
            # plt.xlim(xmin=0)
            pass
        plt.ylim(bottom=0)
        if self.plot_type == PlotType.L_GRAPH or self.plot_type == PlotType.HIST:
            plt.ylim(top=self.y_max)
            plt.xlim(right=self.x_max)
        plt.xlabel(x_label, fontweight="bold", fontsize=14)
        plt.ylabel(y_label, fontweight="bold", fontsize=14)

        if self.yscale:
            plt.yscale(self.yscale)

        plt.xticks(weight="bold", fontsize=14)
        plt.yticks(weight="bold", fontsize=14)
        plt.grid()
        ax = plt.gca()
        if not (self.plot_type == PlotType.HIST):
            ax.xaxis.set_major_formatter(default_major_formatter)
        ax.yaxis.set_major_formatter(default_major_formatter)
        if self.plot_type == PlotType.L_GRAPH or self.plot_type == PlotType.HIST:
            ax.yaxis.set_major_formatter(sec_major_formatter)

        plot_directory = self._make_plot_directory()
        for x in ["pdf", "png"]:
            filename = os.path.join(plot_directory, f"{self.plot_name}.{x}")
            plt.savefig(filename, bbox_inches="tight", dpi=300)


def overhead():
    transaction_size = 512

    plotter = Plotter(
        "overhead",
        PlotType.L_GRAPH,
        y_max=1,
        x_max=55000,
        legend_columns=3,
        legend_location="upper center",
        legend_anchor=(0.5, 1),
        figure_size=(6.4, 2.4),
        frameon=True,
    )

    plotter.new_system()
    data = MysticetiDataNew("mysticeti-c-new", 10, 0, transaction_size, "shared")
    plotter.plot(data.for_latency_throughput_plot())

    data = MysticetiDataNew("mysticeti-c-new", 50, 0, transaction_size, "shared")
    plotter.plot(data.for_latency_throughput_plot())

    plotter.new_system()
    data = ObeliaData("obelia", 0, 10, 50, transaction_size, "shared")
    plotter.plot(data.for_latency_throughput_plot())

    data = ObeliaData("obelia", 0, 50, 200, transaction_size, "shared")
    plotter.plot(data.for_latency_throughput_plot())

    plotter.new_system()
    data = ObeliaData("obelia", 40, 10, 50, transaction_size, "shared")
    plotter.plot(data.for_latency_throughput_plot())

    data = ObeliaData("obelia", 190, 50, 200, transaction_size, "shared")
    plotter.plot(data.for_latency_throughput_plot())

    plotter.finalize()


def histogram():
    transaction_size = 512

    plotter = Plotter(
        "histogram",
        PlotType.HIST,
        y_max=1,
        legend_columns=3,
        legend_location="upper center",
        legend_anchor=(0.5, 1),
        figure_size=(6.4, 2.4),
    )

    # Ugly incremental spacing hack in labels to tell matplotlib that they are
    # different labels

    plotter.new_system()
    data = MysticetiDataNew("mysticeti-c-new", 10, 0, transaction_size, "shared")
    plotter.bar(data.for_histogram("M", 50000))

    data = MysticetiDataNew("mysticeti-c-new", 50, 0, transaction_size, "shared")
    plotter.bar(data.for_histogram(" M ", 50000))

    plotter.new_system()
    data = ObeliaData("obelia", 0, 10, 50, transaction_size, "shared")
    plotter.bar(data.for_histogram("O", 50000))

    data = ObeliaData("obelia", 0, 50, 200, transaction_size, "shared")
    plotter.bar(data.for_histogram(" O ", 50000))

    plotter.new_system()
    data = ObeliaData("obelia", 40, 10, 50, transaction_size, "shared")
    plotter.bar(data.for_histogram("  O  ", 50000))

    data = ObeliaData("obelia", 190, 50, 200, transaction_size, "shared")
    plotter.bar(data.for_histogram("   O   ", 50000))

    plotter.finalize()


if __name__ == "__main__":
    overhead()
    histogram()
