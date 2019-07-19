# Copyright 2018-2019 Ben Mather, Robert Delhaye
#
# This file is part of PyCurious.
#
# PyCurious is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# PyCurious is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PyCurious.  If not, see <http://www.gnu.org/licenses/>.

"""
This module provides some simple functions to download or install files.

It is good practise to store a checksum for each file. A checksum is a
unique signature used to identify a file, and ensure that the data
contained within the file is legitimate. Here is a sample workflow for
downloading EMAG2 (v3) and evaluating its checksum:

```python
resource = {
     "local_file":"../../data/EMAG2_V3_20170530.npz",
     "md5":'c0898b6a91efb3f13783873a8b67380c',
     "url":"https://zenodo.org/record/3245551/files/EMAG2_V3_20170530.npz?download=1",
     "expected_size":"500Mb"
    }

download_cached_file(resource["url"], resource["local_file"], resource["md5"], resource["expected_size"])
```

The file will commence downloading if it does not already exist in the local directory
or if the checksum (md5) does not match. If you do not know the checksum for a file, run

```python
md5sum(filename)
```

to return its unique identifier.
"""


def download_file(url, local_filename, expected_size="Unknown"):
    """
    Download files from a URL to a local filename.

    Args:
        url : str
            URL that points to the file to be downloaded
        local_filename : str
            download content to this filename
    """
    import requests
    import os
    import time
    import shutil

    # We might want to bundle some files if they are small / compressed or not readily available for download

    if os.path.isfile(url):
        shutil.copy(url, local_filename)

    else:
        r = requests.get(url, stream=True)

        start_time = time.time()
        last_time = start_time
        datasize = 0

        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=10000000):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
                    datasize += len(chunk)
                    if (time.time() - last_time) > 2.5:
                        print(
                            "{:.1f} Mb in {:.1f}s / {}".format(
                                datasize / 1.0e6,
                                time.time() - start_time,
                                expected_size,
                            )
                        )
                        last_time = time.time()

    return


def md5sum(filename):
    """
    Returns the checksum for a given file
    """
    import hashlib
    from functools import partial

    with open(filename, mode="rb") as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 4096), b""):
            d.update(buf)
    return d.hexdigest()


def download_cached_file(
    location_url, local_file, expected_md5, expected_size="Unknown"
):
    """
    Download files from a URL to a local filename.

    Same as `download_file()` but checks whether the file has already
    been downloaded from its checksum.

    Args:
        location_url : str
            URL that points to the file to be downloaded
        local_file : str
            download content to this filename
        expected_md5 : str
            checksum belonging to that file
        expected_size : float / int / str (optional)
            optional size of file (default="Unknown")
    """
    import sys

    try:
        assert md5sum(local_file) == expected_md5
        print("Using cached file - {}".format(local_file))
        return 2

    except (IOError, AssertionError) as error_info:
        # No file or the wrong file ... best go download it
        # print "Assertion failed - {}".format(sys.exc_info())

        try:
            data_file = download_file(location_url, local_file, expected_size)
            print("Downloaded from {}".format(location_url))
            return 1

        except:
            print("Unable to download {} [{}] ".format(location_url, sys.exc_info()))
            return 0


def report_cached_file(local_file, expected_md5):
    """
    Report whether the local file matches its checksum

    Args:
        local_file : str
            string pointing to the local file
        expected_md5 : str
            checksum assigned to `local_file`.
    """
    import os
    import os.path

    if not os.path.isfile(local_file):
        print("Local file {} does not exist".format(local_file))
    else:
        if len(expected_md5) == 0 or md5sum(local_file) == expected_md5:
            print("Cached file {} is valid".format(local_file))
        else:
            print(
                "Cached file {} failed, checksum {}".format(
                    local_file, md5sum(local_file)
                )
            )
