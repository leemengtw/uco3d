# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import requests
import functools
import json
import warnings
import time
import random
import hashlib
import copy
import subprocess
import traceback

from typing import List, Optional
from multiprocessing import Pool, Lock
from multiprocessing.dummy import Pool as SerialPool
from tqdm import tqdm

# import os
# import json
# import boto3  # 用於 S3 操作
# from botocore.exceptions import NoCredentialsError
# import requests


BLOCKSIZE = 65536  # for sha256 computation
lock = Lock()

def ensure_permissions():
    os.setuid(os.getuid())
    os.setgid(os.getgid())


DEFAULT_DOWNLOAD_MODALITIES = [
    "metadata",
    # "depth_maps",   # by default we do not download depth maps!
    "rgb_videos",
    "mask_videos",
    "gaussian_splats",
    "point_clouds",
    "sparse_point_clouds",
    "segmented_point_clouds",
]

from boto3.s3.transfer import TransferConfig
import boto3

s3_client = boto3.client("s3")

config = TransferConfig(multipart_threshold=1024 * 25,
                        multipart_chunksize=1024 * 25,
                        use_threads=True)

def upload_to_s3(
    local_file_path, 
    bucket_name="mod3d-west"
):

    try:
        s3_key = os.path.join("uco3d", os.path.basename(local_file_path))
    
        print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
        s3_client.upload_file(local_file_path, bucket_name, s3_key, Config=config)
        print(f"File successfully uploaded to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file: {e}")


def download_dataset(
    category_to_archives_file: str,
    link_list_file: str,
    download_folder: str,
    n_download_workers: int = 4,
    n_extract_workers: int = 4,
    download_small_subset: bool = False,
    download_super_categories: Optional[List[str]] = None,
    download_modalities: Optional[List[str]] = DEFAULT_DOWNLOAD_MODALITIES,
    checksum_check: bool = False,
    clear_archives_after_unpacking: bool = False,
    skip_downloaded_archives: bool = True,
    crash_on_checksum_mismatch: bool = False,
    debug = False,
    existing_s3_files_txt_file_path = None,
):
    """
    Downloads and unpacks the dataset in UCO3D format.

    Note: The script will make a folder `<download_folder>/_in_progress`, which
        stores files whose download is in progress. The folder can be safely deleted
        once the download is finished.

    Args:
        link_list_file: A text file with the list of zip file download links.
        download_folder: A local target folder for downloading the
            the dataset files.
        n_download_workers: The number of parallel workers
            for downloading the dataset files.
        n_extract_workers: The number of parallel workers
            for extracting the dataset files.
        download_small_subset: Download only a small debug subset of 52 videos with
            including all available modalities and supercategories.
            As such, cannot be used together with setting
            `download_super_categories` or `download_modalities`.
        download_super_categories: A list of super categories to download.
            If `None`, downloads all.
        download_modalities: A list of modalities to download.
            If `None`, downloads all.
        checksum_check: Enable validation of the downloaded file's checksum before
            extraction.
        clear_archives_after_unpacking: Delete the unnecessary downloaded archive files
            after unpacking.
        skip_downloaded_archives: Skip re-downloading already downloaded archives.
        crash_on_checksum_mismatch: Crashes the script if the checksums of downloaded
            files do not match the expected ones.
    """

    if not os.path.isdir(download_folder):
        raise ValueError(
            "Please specify `download_folder` with a valid path to a target folder"
            + " for downloading the dataset."
            + f" {download_folder} does not exist."
        )

    if link_list_file.startswith("http"):
        # download the link list file
        print(f"Downloading link list file {link_list_file}.")
        link_list_file_local = os.path.join(
            download_folder, "uco3d_dataset_download_urls.json"
        )
        _download_with_progress_bar(
            link_list_file, link_list_file_local, "uco3d_dataset_download_urls.json",
            quiet=True,
        )
        link_list_file = link_list_file_local

    elif not os.path.isfile(link_list_file):
        raise ValueError(
            "Please specify `link_list_file` with a valid path to a json"
            " with download links to the uco3d zip files."
        )

    if not os.path.isfile(category_to_archives_file):
        raise ValueError(
            "Please specify `category_to_archives_file` with a valid path to a json"
            " with mapping between dataset categories and archive filenames."
        )

    if download_small_subset:
        if download_super_categories is not None:
            raise ValueError(
                "The `download_small_subset` flag cannot be used together with"
                + " `download_super_categories`."
            )
        if (download_modalities is not None) and (
            set(download_modalities) != set(DEFAULT_DOWNLOAD_MODALITIES)
        ):
            warnings.warn(
                "The `download_small_subset` flag is set, but `download_modalities`"
                + " is not None or does not match the default modalities."
                + " The `download_modalities` flag will be ignored."
            )

    # read the links file
    with open(link_list_file, "r") as f:
        links: dict = json.load(f)["main_data"]

    with open(category_to_archives_file, "r") as f:
        category_to_archives: dict = json.load(f)

    if debug:
        print("DEBUGGING")

        # remove_s3_prefix("s3://mod3d-west/uco3d/_in_progress")

        target_modalities = ['rgb_videos']
        target_link_name = "part_rgb_videos_0460.zip"

        links = {fname: info for fname, info in links.items() if fname == target_link_name}
        download_modalities = target_modalities
        category_to_archives = {k: v for k, v in category_to_archives.items() if k in target_modalities}
        
        print(f"{links=}")

    # extract possible modalities, super categories
    uco3d_modalities = set()
    uco3d_super_categories = set()
    for modality, modality_links in category_to_archives.items():
        uco3d_modalities.add(modality)
        if modality == "metadata":
            continue
        for super_category, super_category_links in modality_links.items():
            uco3d_super_categories.add(super_category)

    # check if the requested super_categories, or modalities are valid
    for sel_name, download_sel, possible in zip(
        ("super_category", "modality"),
        (download_super_categories, download_modalities),
        (uco3d_super_categories, uco3d_modalities),
    ):
        if download_sel is not None:
            for sel in download_sel:
                if sel not in possible:
                    raise ValueError(
                        f"Invalid choice for '{sel_name}': {sel}. "
                        + f"Possible choices are: {str(possible)}."
                    )

    def _is_for_download(
        modality: str,
        super_category: str,
    ) -> bool:
        if download_modalities is not None and modality not in download_modalities:
            return False
        if download_super_categories is None:
            return True
        if super_category in download_super_categories:
            return True
        return False

    def _add_to_data_links(data_links, link_data):
        # copy the link data and replace the filename with the actual link
        link_data_with_link = copy.deepcopy(link_data)
        link_data_with_link["download_url"] = links[link_data["filename"]][
            "download_url"
        ]
        data_links.append(link_data_with_link)

    # determine links to files we want to download
    data_links = []
    if download_small_subset:
        _add_to_data_links(data_links, category_to_archives["examples"])
    else:
        actual_download_supercategories_modalities = set()
        for modality, modality_links in category_to_archives.items():
            if modality == "metadata":
                assert isinstance(modality_links, dict)
                _add_to_data_links(data_links, modality_links)
                continue
            for super_category, super_category_links in modality_links.items():
                if _is_for_download(modality, super_category):
                    actual_download_supercategories_modalities.add(
                        f"{modality}/{super_category}"
                    )
                    for link_name, link_data in super_category_links.items():
                        if debug:
                            if link_name == target_link_name:
                                _add_to_data_links(data_links, link_data)
                        else:
                            _add_to_data_links(data_links, link_data)

    for modality_super_category in sorted(
        actual_download_supercategories_modalities
    ):
        print(f"Downloading {modality_super_category}.")


    # skip existing s3 files
    existing_link_names = []
    if existing_s3_files_txt_file_path:
        
        command = ["aws", "s3", "ls", "s3://mod3d-west/uco3d/"]

        with open(existing_s3_files_txt_file_path, "w") as file:
            subprocess.run(command, stdout=file, check=True)

        with open(existing_s3_files_txt_file_path, "r") as fp:
            lines = fp.readlines()

        for line in lines:
            link_names = line.split(" ")[-1].strip()
            existing_link_names.append(link_names)
            print(f"{link_names} already available on S3. skip")

    print(f"Skip {len(existing_link_names)} files")
    data_links = [d for d in data_links if not d["filename"] in existing_link_names]
    print(f"{len(data_links)} files waiting to be downloaded")


    # multiprocessing pool
    with _get_pool_fn(n_download_workers)(
        processes=n_download_workers
    ) as download_pool:
        print(f"Downloading {len(data_links)} dataset files ...")
        download_ok = {}
        for link_name, ok in tqdm(
            download_pool.imap(
                functools.partial(
                    _download_file,
                    download_folder,
                    checksum_check,
                    skip_downloaded_archives,
                    crash_on_checksum_mismatch,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            download_ok[link_name] = ok

        if not all(download_ok.values()):
            not_ok_links = [n for n, ok in download_ok.items() if not ok]
            not_ok_links_str = "\n".join(not_ok_links)
            raise AssertionError(
                "The SHA256 checksums did not match for some of the downloaded files:\n"
                + not_ok_links_str
                + "\n"
                + "This is most likely due to a network failure."
                + " Please restart the download script."
            )

    # print(f"Extracting {len(data_links)} dataset files ...")
    # with _get_pool_fn(n_extract_workers)(processes=n_extract_workers) as extract_pool:
    #     for _ in tqdm(
    #         extract_pool.imap(
    #             functools.partial(
    #                 _unpack_file,
    #                 download_folder,
    #                 clear_archives_after_unpacking,
    #             ),
    #             data_links,
    #         ),
    #         total=len(data_links),
    #     ):
    #         pass

    # clean up the in-progress folder if empty
    in_progress_folder = _get_in_progress_folder(download_folder)
    if os.path.isdir(in_progress_folder) and len(os.listdir(in_progress_folder)) == 0:
        print(f"Removing in-progress downloads folder {in_progress_folder}")
        shutil.rmtree(in_progress_folder)

    print("Done")


def _sha256_file(path: str):
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        file_buffer = f.read(BLOCKSIZE)
        while len(file_buffer) > 0:
            sha256_hash.update(file_buffer)
            file_buffer = f.read(BLOCKSIZE)
    digest_ = sha256_hash.hexdigest()
    return digest_


def _get_in_progress_folder(download_folder: str):
    return os.path.join(download_folder, "_in_progress")


def _get_pool_fn(n_workers: int):
    if n_workers <= 1:
        return SerialPool
    return Pool


def _unpack_file(
    download_folder: str,
    clear_archive: bool,
    link_data: dict,
):
    link_name = link_data["filename"]
    local_fl = os.path.join(download_folder, link_name)
    print(f"Unpacking dataset file {local_fl} ({link_name}) to {download_folder}.")
    # important, shutil.unpack_archive is not thread-safe:
    time.sleep(random.random() * 0.3)
    shutil.unpack_archive(local_fl, download_folder)
    if clear_archive:
        os.remove(local_fl)


import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# 配置 S3
S3_BUCKET = "mod3d-west"

s3_client = boto3.client("s3")


def move_s3_file(source_key, destination_key):
    """
    在 S3 中移動文件，對於大於 5GB 的文件使用多部分拷貝。

    :param source_key: 原文件的 S3 key
    :param destination_key: 目標文件的 S3 key
    """
    try:
        # 檢查文件大小
        response = s3_client.head_object(Bucket=S3_BUCKET, Key=source_key)
        file_size = response["ContentLength"]

        if file_size > 5 * 1024 * 1024 * 1024:  # 超過 5GB
            print(f"File size {file_size} exceeds 5GB, using multipart copy...")
            multipart_copy(source_key, destination_key)
        else:
            # 標準拷貝
            print(f"Copying {source_key} to {destination_key} using standard copy...")
            s3_client.copy_object(
                Bucket=S3_BUCKET,
                CopySource={"Bucket": S3_BUCKET, "Key": source_key},
                Key=destination_key
            )
            print(f"Copied {source_key} to {destination_key}")

        # 刪除原文件
        # print(f"Deleting original file {source_key}...")
        # s3_client.delete_object(Bucket=S3_BUCKET, Key=source_key)
        # print(f"Deleted {source_key}")
    except NoCredentialsError:
        print("AWS credentials not found. Please configure your AWS credentials.")
        raise
    except ClientError as e:
        print(f"Error occurred: {e}")
        print(f"{source_key=}")
        print(f"{destination_key=}")
        raise


def multipart_copy(source_key, destination_key):
    """
    使用多部分上傳方式在 S3 中複製大於 5GB 的文件。

    :param source_key: 源文件的 S3 Key
    :param destination_key: 目標文件的 S3 Key
    """
    try:
        # 獲取源文件的大小
        response = s3_client.head_object(Bucket=S3_BUCKET, Key=source_key)
        source_size = response["ContentLength"]
        print(f"Source file size: {source_size} bytes")

        # 初始化多部分上傳
        multipart_upload = s3_client.create_multipart_upload(Bucket=S3_BUCKET, Key=destination_key)
        upload_id = multipart_upload["UploadId"]

        try:
            # 設置每部分的大小（例如 500MB）
            part_size = 500 * 1024 * 1024
            part_number = 1
            parts = []

            for start in range(0, source_size, part_size):
                end = min(start + part_size - 1, source_size - 1)
                print(f"Copying bytes {start}-{end}...")

                # 複製每部分
                part = s3_client.upload_part_copy(
                    Bucket=S3_BUCKET,
                    Key=destination_key,
                    CopySource={"Bucket": S3_BUCKET, "Key": source_key},
                    CopySourceRange=f"bytes={start}-{end}",
                    PartNumber=part_number,
                    UploadId=upload_id,
                )
                parts.append({"ETag": part["CopyPartResult"]["ETag"], "PartNumber": part_number})
                part_number += 1

            # 完成多部分上傳
            s3_client.complete_multipart_upload(
                Bucket=S3_BUCKET,
                Key=destination_key,
                MultipartUpload={"Parts": parts},
                UploadId=upload_id,
            )
            print(f"Successfully copied {source_key} to {destination_key} using multipart upload.")
        except Exception as e:
            # 如果出錯，取消多部分上傳
            s3_client.abort_multipart_upload(Bucket=S3_BUCKET, Key=destination_key, UploadId=upload_id)
            print(f"Multipart upload aborted due to: {e}")
            raise

    except NoCredentialsError:
        print("AWS credentials not found. Please configure your AWS credentials.")
        raise
    except ClientError as e:
        print(f"Error occurred: {e}")
        raise


# def remove_s3_file(source_key):
#     """
#     在 S3 中移動文件。

#     :param source_key: 原文件的 S3 key
#     :param destination_key: 目標文件的 S3 key
#     """
#     try:
#         # 刪除原文件
#         print(f"Deleting original file {source_key}...")
#         s3_client.delete_object(Bucket=S3_BUCKET, Key=source_key)
#         print(f"Deleted {source_key}")
#     except NoCredentialsError:
#         print("AWS credentials not found. Please configure your AWS credentials.")
#     except ClientError as e:
#         print(f"Error occurred: {e}")
#         print(f"{source_key=}")


def remove_s3_prefix(prefix):
    try:
        continuation_token = None
        while True:
            if continuation_token:
                response = s3_client.list_objects_v2(
                    Bucket=S3_BUCKET, Prefix=prefix, ContinuationToken=continuation_token
                )
            else:
                response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

            if "Contents" in response:
                for obj in response["Contents"]:
                    file_key = obj["Key"]
                    print(f"Deleting file: {file_key}...")
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=file_key)
                    print(f"Deleted {file_key}")

            # 檢查是否有下一頁
            if response.get("IsTruncated"):  # 還有下一頁
                continuation_token = response["NextContinuationToken"]
            else:  # 已到最後一頁
                break

        print(f"All files under prefix '{prefix}' have been deleted.")
    except NoCredentialsError:
        print("AWS credentials not found. Please configure your AWS credentials.")
    except ClientError as e:
        print(f"Error occurred: {e}")


def _download_file(
    download_folder: str,
    checksum_check: bool,
    skip_downloaded_files: bool,
    crash_on_checksum_mismatch: bool,
    link_data: dict,
):
    ensure_permissions()

    url = link_data["download_url"]
    link_name = link_data["filename"]
    sha256 = link_data["sha256sum"]
    local_fl_final = os.path.join(download_folder, link_name)

    if skip_downloaded_files:
        if os.path.isfile(local_fl_final):
            print(f"Skipping {local_fl_final}, already downloaded!")
            return link_name, True
        # elif 

    in_progress_folder = _get_in_progress_folder(download_folder)
    os.makedirs(in_progress_folder, exist_ok=True)
    local_fl = os.path.join(in_progress_folder, link_name)

    print(f"Downloading dataset file {link_name} ({url}) to {local_fl}.")
    num_max_retries = 3
    num_tries = 0
    success = False

    while not success:
        success = _download_with_progress_bar(url, local_fl, link_name)
        num_tries += 1
        if num_tries >= num_max_retries:
            return link_name, False

    if checksum_check:
        print(f"Checking SHA256 for {local_fl}.")
        sha256_local = _sha256_file(local_fl)
        if sha256_local != sha256:
            msg = (
                f"Checksums for {local_fl} did not match!"
                + " This is likely due to a network failure,"
                + " please restart the download script."
                + f" Expected: {sha256}, got: {sha256_local}."
            )
            if crash_on_checksum_mismatch:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

            # s3_fl = os.path.abspath(slocal_fl).replace("/admin/home-meng/s3_mount/mod3d-west/", "")
            # remove_s3_file(s3_fl)

            return link_name, False
        
    os.rename(local_fl, local_fl_final)

    upload_to_s3(local_fl_final)

    # try:
    #     os.rename(local_fl, local_fl_final)
    # except FileNotFoundError:
    #     print(f"{local_fl} not available yet, skipped")
    #     return link_name, False

    # shutil.move(local_fl, local_fl_final)

    # /admin/home-meng/s3_mount/mod3d-west/uco3d/_in_progress/part_gaussian_splats_0001.zip
    # /admin/home-meng/s3_mount/mod3d-west/uco3d/part_gaussian_splats_0001.zip
    # s3_fl = os.path.abspath(local_fl).replace("/admin/home-meng/s3_mount/mod3d-west/", "")
    # s3_fl_final = os.path.abspath(local_fl_final).replace("/admin/home-meng/s3_mount/mod3d-west/", "")

    # print(f"{s3_fl=}")
    # print(f"{s3_fl_final=}")
    # move_s3_file(s3_fl, s3_fl_final)

    return link_name, True


def write_file_locally_and_move(fname, resp, total, quiet, filename):

    temp_path = f"/tmp/{os.path.basename(fname)}"

    with open(temp_path, "wb") as file, tqdm(
            desc=temp_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for datai, data in enumerate(resp.iter_content(chunk_size=1024)):
                size = file.write(data)
                bar.update(size)
                if (not quiet) and (datai % max((max(total // 1024, 1) // 20), 1) == 0):
                    print(
                        f"{filename}: Downloaded {100.0*(float(bar.n)/max(total, 1)):3.1f}%."
                    )
                    print(bar)

    print(f"Copy {temp_path} to {fname}")
    shutil.move(temp_path, fname)


def _download_with_progress_bar(url: str, fname: str, filename: str, quiet: bool = False):
    # taken from https://stackoverflow.com/a/62113293/986477
    if not url.startswith("http"):
        # url is in fact a local path, so we copy to the download folder
        print(f"Local copy {url} -> {fname}")
        shutil.copy(url, fname)
        return
    resp = requests.get(url, stream=True)
    print(url)
    total = int(resp.headers.get("content-length", 0))
    # write_file_locally_and_move(fname, resp=resp, total=total, quiet=quiet, filename=filename)

    try:
        with open(fname, "wb") as file, tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for datai, data in enumerate(resp.iter_content(chunk_size=1024)):
                size = file.write(data)
                bar.update(size)
                if (not quiet) and (datai % max((max(total // 1024, 1) // 20), 1) == 0):
                    print(
                        f"{filename}: Downloaded {100.0*(float(bar.n)/max(total, 1)):3.1f}%."
                    )
                    print(bar)
    except Exception as e:
        print(f"Encountered issue when downloading {fname}. probably not able to update partial downloaded file. remove existing file if present")
        print(traceback.format_exc())
        print()

        # s3_fl = os.path.abspath(fname).replace("/admin/home-meng/s3_mount/mod3d-west/", "")
        # remove_s3_file(s3_fl)

        return False
    
    return True

