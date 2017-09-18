# Copyright (C) 2017 University of Coimbra
#
# This program is free software: you can redistribute it and/or modify
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
#
# Original author and maintainer: Gonçalo S. Martins (gondsm@gmail.com)

# This needs to be sourced in order for the ProBT-based modules to work when testing with non-ROS system.
PL_DIR="/home/vsantos/Desktop/probt-spl-3.0.0-linux64-dynamic-release/lib/"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PL_DIR}

export PYTHONPATH=${PL_DIR}:${PYTHONPATH} "$@"