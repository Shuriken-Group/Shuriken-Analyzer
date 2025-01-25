#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from shuriken import *

if __name__ == "__main__":
    path = "../../../tests/compiled/"
    for file in os.listdir(path):
        if not file.endswith(".apk"):
            continue

        apk = Apk(os.path.join(path, file), True)

        print(
            f"Number of method analysis objects: {apk.get_number_of_methodanalysis_objects()}"
        )

        for j in range(apk.get_number_of_methodanalysis_objects()):
            method_analysis: hdvmmethodanalysis_t = apk.get_analyzed_method_by_idx(j)
            print(f"{j} Method name: {method_analysis.full_name.decode()}")
            print(f"\tIs external: {method_analysis.external}")
