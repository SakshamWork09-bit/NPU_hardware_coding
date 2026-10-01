import ctypes
import csv
import os
import time
from ctypes import wintypes


# ============================================================
# HWiNFO SHARED MEMORY TELEMETRY
# MotoBook 60 Pro / Core Ultra 5 225H / Arc 130T / Intel NPU
# ============================================================

SHM_NAME = r"Global\HWiNFO_SENS_SM2"
MUTEX_NAME = r"Global\HWiNFO_SM2_MUTEX"

FILE_MAP_READ = 0x0004

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

OpenFileMappingW = kernel32.OpenFileMappingW
OpenFileMappingW.argtypes = [
    wintypes.DWORD,
    wintypes.BOOL,
    wintypes.LPCWSTR
]
OpenFileMappingW.restype = wintypes.HANDLE

MapViewOfFile = kernel32.MapViewOfFile
MapViewOfFile.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_size_t
]
MapViewOfFile.restype = ctypes.c_void_p

UnmapViewOfFile = kernel32.UnmapViewOfFile
UnmapViewOfFile.argtypes = [ctypes.c_void_p]

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]

OpenMutexW = kernel32.OpenMutexW
OpenMutexW.argtypes = [
    wintypes.DWORD,
    wintypes.BOOL,
    wintypes.LPCWSTR
]
OpenMutexW.restype = wintypes.HANDLE

WaitForSingleObject = kernel32.WaitForSingleObject
WaitForSingleObject.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD
]
WaitForSingleObject.restype = wintypes.DWORD

ReleaseMutex = kernel32.ReleaseMutex
ReleaseMutex.argtypes = [wintypes.HANDLE]


HWINFO_MAGIC = int.from_bytes(b"HWiS", "little")


def c_string(ptr, length):
    raw = ctypes.string_at(ptr, length)
    return raw.split(b"\x00", 1)[0].decode(
        "utf-8",
        errors="replace"
    ).strip()


class HWiNFO:

    def __init__(self):
        self.h_map = None
        self.view = None
        self.mutex = None

    def connect(self):

        self.h_map = OpenFileMappingW(
            FILE_MAP_READ,
            False,
            SHM_NAME
        )

        if not self.h_map:
            raise RuntimeError(
                "Cannot open HWiNFO shared memory.\n\n"
                "Check:\n"
                "1. HWiNFO64 is running\n"
                "2. Sensors window is open\n"
                "3. Shared Memory Support is enabled"
            )

        self.view = MapViewOfFile(
            self.h_map,
            FILE_MAP_READ,
            0,
            0,
            0
        )

        if not self.view:
            self.close()
            raise RuntimeError("MapViewOfFile failed.")

        self.mutex = OpenMutexW(
            0x001F0001,
            False,
            MUTEX_NAME
        )

        magic = ctypes.c_uint32.from_address(
            self.view
        ).value

        if magic != HWINFO_MAGIC:
            self.close()
            raise RuntimeError(
                f"Invalid HWiNFO shared-memory signature: "
                f"{magic:08X}"
            )

    def close(self):

        if self.view:
            UnmapViewOfFile(self.view)
            self.view = None

        if self.h_map:
            CloseHandle(self.h_map)
            self.h_map = None

        if self.mutex:
            CloseHandle(self.mutex)
            self.mutex = None

    def snapshot(self):

        if self.mutex:
            result = WaitForSingleObject(
                self.mutex,
                1000
            )

            if result not in (0, 0x80):
                raise RuntimeError(
                    "HWiNFO shared-memory mutex timeout."
                )

        try:

            base = self.view

            # ------------------------------------------------
            # HEADER
            # ------------------------------------------------

            b = bytes(
                ctypes.cast(
                    base,
                    ctypes.POINTER(
                        ctypes.c_uint8 * 56
                    )
                ).contents
            )

            signature = int.from_bytes(
                b[0:4], "little"
            )

            if signature != HWINFO_MAGIC:
                raise RuntimeError(
                    "HWiNFO shared memory signature changed."
                )

            sensor_offset = int.from_bytes(
                b[20:24], "little"
            )

            sensor_size = int.from_bytes(
                b[24:28], "little"
            )

            sensor_count = int.from_bytes(
                b[28:32], "little"
            )

            reading_offset = int.from_bytes(
                b[32:36], "little"
            )

            reading_size = int.from_bytes(
                b[36:40], "little"
            )

            reading_count = int.from_bytes(
                b[40:44], "little"
            )

            # ------------------------------------------------
            # SENSORS
            # ------------------------------------------------

            sensors = {}

            for i in range(sensor_count):

                ptr = (
                    base
                    + sensor_offset
                    + i * sensor_size
                )

                sensor_id = ctypes.c_uint32.from_address(
                    ptr
                ).value

                instance = ctypes.c_uint32.from_address(
                    ptr + 4
                ).value

                original = c_string(
                    ptr + 8,
                    128
                )

                user = c_string(
                    ptr + 136,
                    128
                )

                sensors[i] = {
                    "id": sensor_id,
                    "instance": instance,
                    "name": user or original,
                    "original": original
                }

            # ------------------------------------------------
            # READINGS
            # ------------------------------------------------

            readings = []

            for i in range(reading_count):

                ptr = (
                    base
                    + reading_offset
                    + i * reading_size
                )

                reading_type = ctypes.c_uint32.from_address(
                    ptr
                ).value

                sensor_index = ctypes.c_uint32.from_address(
                    ptr + 4
                ).value

                reading_id = ctypes.c_uint32.from_address(
                    ptr + 8
                ).value

                label_original = c_string(
                    ptr + 12,
                    128
                )

                label_user = c_string(
                    ptr + 140,
                    128
                )

                unit = c_string(
                    ptr + 268,
                    16
                )

                value = ctypes.c_double.from_address(
                    ptr + 284
                ).value

                minimum = ctypes.c_double.from_address(
                    ptr + 292
                ).value

                maximum = ctypes.c_double.from_address(
                    ptr + 300
                ).value

                average = ctypes.c_double.from_address(
                    ptr + 308
                ).value

                sensor = sensors.get(sensor_index)

                if sensor is None:
                    continue

                readings.append({
                    "sensor": sensor["name"],
                    "label": label_user or label_original,
                    "unit": unit,
                    "value": value,
                    "min": minimum,
                    "max": maximum,
                    "avg": average,
                    "type": reading_type,
                    "id": reading_id
                })

            return readings

        finally:

            if self.mutex:
                ReleaseMutex(self.mutex)


# ============================================================
# FIND SENSOR
# ============================================================

def get(readings, sensor, label):

    sensor = sensor.lower()
    label = label.lower()

    for r in readings:

        if (
            sensor in r["sensor"].lower()
            and
            label == r["label"].lower()
        ):
            return r["value"]

    return None


def get_contains(readings, sensor, label):

    sensor = sensor.lower()
    label = label.lower()

    for r in readings:

        if (
            sensor in r["sensor"].lower()
            and
            label in r["label"].lower()
        ):
            return r["value"]

    return None


def cpu_limit(readings, label):

    return get_contains(
        readings,
        "Performance Limit Reasons",
        label
    )


# ============================================================
# EXTRACT EXACT TELEMETRY
# ============================================================

def extract(readings):

    row = {}

    # ========================================================
    # CPU
    # ========================================================

    row["CPU_Temp_C"] = get(
        readings,
        "Enhanced",
        "CPU Package"
    )

    row["CPU_Package_Power_W"] = get(
        readings,
        "Enhanced",
        "CPU Package Power"
    )

    row["CPU_Usage_%"] = get(
        readings,
        "Enhanced",
        "Total CPU Usage"
    )

    row["CPU_Effective_Clock_MHz"] = get(
        readings,
        "Enhanced",
        "Core Effective Clock"
    )

    # CPU thermal throttling
    row["CPU_Thermal_Throttle"] = get(
        readings,
        "DTS",
        "Package/Ring Thermal Throttling"
    )

    # CPU package power limit
    row["CPU_Package_Power_Limit"] = get(
        readings,
        "DTS",
        "Package/Ring Power Limit Exceeded"
    )

    # Actual performance-limit reasons
    row["CPU_PL1_Limit"] = cpu_limit(
        readings,
        "IA: Package-Level RAPL/PBM PL1"
    )

    row["CPU_PL2_Limit"] = cpu_limit(
        readings,
        "IA: Package-Level RAPL/PBM PL2,PL3"
    )

    row["CPU_Max_Turbo_Limit"] = cpu_limit(
        readings,
        "IA: Max Turbo Limit"
    )

    row["CPU_Thermal_Event"] = cpu_limit(
        readings,
        "IA: Thermal Event"
    )

    row["CPU_PROCHOT"] = cpu_limit(
        readings,
        "IA: PROCHOT"
    )

    row["CPU_VR_Thermal"] = cpu_limit(
        readings,
        "IA: VR Thermal Alert"
    )

    # ========================================================
    # GPU
    # ========================================================

    row["GPU_Temp_C"] = get(
        readings,
        "Intel Arc 130T",
        "GPU Core Temperature"
    )

    row["GPU_Power_W"] = get(
        readings,
        "Intel Arc 130T",
        "IGPU Power"
    )

    row["GPU_Clock_MHz"] = get(
        readings,
        "Intel Arc 130T",
        "GPU Clock"
    )

    row["GPU_Util_%"] = get(
        readings,
        "Intel Arc 130T",
        "GPU D3D Usage"
    )

    # GPU limit reasons
    row["GPU_PL1_Limit"] = get(
        readings,
        "Intel Arc 130T",
        "Avg. Power (PL1)"
    )

    row["GPU_PL2_Limit"] = get(
        readings,
        "Intel Arc 130T",
        "Burst Power (PL2)"
    )

    row["GPU_PL4_Limit"] = get(
        readings,
        "Intel Arc 130T",
        "Current (PL4)"
    )

    row["GPU_Thermal_Limit"] = get(
        readings,
        "Intel Arc 130T",
        "Thermal"
    )

    row["GPU_Power_Limit"] = get(
        readings,
        "Intel Arc 130T",
        "Power Supply"
    )

    row["GPU_Software_Limit"] = get(
        readings,
        "Intel Arc 130T",
        "Software Limit"
    )

    row["GPU_Hardware_Limit"] = get(
        readings,
        "Intel Arc 130T",
        "Hardware Limit"
    )

    # ========================================================
    # NPU
    # ========================================================

    row["NPU_Clock_MHz"] = get(
        readings,
        "Intel NPU",
        "NPU Clock"
    )

    row["NPU_Util_%"] = get(
        readings,
        "Intel NPU",
        "NPU D3D Usage"
    )

    row["NPU_Power_Limit"] = get(
        readings,
        "Intel NPU",
        "Power"
    )

    row["NPU_Voltage_Limit"] = get(
        readings,
        "Intel NPU",
        "Voltage Limit"
    )

    row["NPU_PL4_Limit"] = get(
        readings,
        "Intel NPU",
        "Current (PL4)"
    )

    row["NPU_Thermal_Limit"] = get(
        readings,
        "Intel NPU",
        "Thermal"
    )

    row["NPU_Utilization_Limit"] = get(
        readings,
        "Intel NPU",
        "Utilization"
    )

    return row


# ============================================================
# CSV
# ============================================================

FIELDS = [
    "Timestamp",
    "Elapsed_s",

    "CPU_Temp_C",
    "CPU_Package_Power_W",
    "CPU_Usage_%",
    "CPU_Effective_Clock_MHz",
    "CPU_Thermal_Throttle",
    "CPU_Package_Power_Limit",
    "CPU_PL1_Limit",
    "CPU_PL2_Limit",
    "CPU_Max_Turbo_Limit",
    "CPU_Thermal_Event",
    "CPU_PROCHOT",
    "CPU_VR_Thermal",

    "GPU_Temp_C",
    "GPU_Power_W",
    "GPU_Clock_MHz",
    "GPU_Util_%",
    "GPU_PL1_Limit",
    "GPU_PL2_Limit",
    "GPU_PL4_Limit",
    "GPU_Thermal_Limit",
    "GPU_Power_Limit",
    "GPU_Software_Limit",
    "GPU_Hardware_Limit",

    "NPU_Clock_MHz",
    "NPU_Util_%",
    "NPU_Power_Limit",
    "NPU_Voltage_Limit",
    "NPU_PL4_Limit",
    "NPU_Thermal_Limit",
    "NPU_Utilization_Limit",
]


# ============================================================
# MAIN LOGGER
# ============================================================

def main():

    hw = HWiNFO()

    try:

        hw.connect()

        output = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "hwinfo_telemetry.csv"
        )

        print()
        print("=" * 78)
        print(" HWiNFO HARDWARE TELEMETRY LOGGER")
        print("=" * 78)
        print()
        print("Connected to HWiNFO Shared Memory.")
        print()
        print(f"CSV: {output}")
        print()
        print("Sampling interval: 0.5 seconds")
        print("Press CTRL+C to stop.")
        print()

        start = time.perf_counter()

        with open(
            output,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=FIELDS
            )

            writer.writeheader()

            while True:

                now = time.perf_counter()

                readings = hw.snapshot()

                values = extract(readings)

                row = {
                    "Timestamp":
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    "Elapsed_s":
                        round(
                            now - start,
                            3
                        ),
                    **values
                }

                writer.writerow(row)
                f.flush()

                def fmt(v, suffix=""):
                    if v is None:
                        return "N/A"
                    if isinstance(v, float):
                        return f"{v:.2f}{suffix}"
                    return f"{v}{suffix}"

                print(
                    "\r"
                    f"CPU "
                    f"{fmt(values['CPU_Temp_C'], '°C')} "
                    f"{fmt(values['CPU_Package_Power_W'], 'W')} | "
                    f"GPU "
                    f"{fmt(values['GPU_Temp_C'], '°C')} "
                    f"{fmt(values['GPU_Power_W'], 'W')} "
                    f"{fmt(values['GPU_Clock_MHz'], 'MHz')} | "
                    f"NPU "
                    f"{fmt(values['NPU_Clock_MHz'], 'MHz')}",
                    end="",
                    flush=True
                )

                time.sleep(0.5)

    except KeyboardInterrupt:

        print()
        print()
        print("Telemetry logging stopped.")

    except Exception as e:

        print()
        print()
        print("ERROR:")
        print(e)

    finally:

        hw.close()


if __name__ == "__main__":
    main()