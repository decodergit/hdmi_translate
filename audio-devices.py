#!/bin/env python3

#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pyaudio

# Получение списка доступных аудиоустройств
p = pyaudio.PyAudio()
device_count = p.get_device_count()
for i in range(device_count):
    device_info = p.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']}")
    # print(device_info)
    