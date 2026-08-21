import os
import re
import torch
import numpy as np
import random
import logging

NODE_CATEGORY = "list-filter"

def extract_first_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else float('inf')

sort_methods = [
    "None",
    "Alphabetical (ASC)",
    "Alphabetical (DESC)",
    "Numerical (ASC)",
    "Numerical (DESC)",
    "Datetime (ASC)",
    "Datetime (DESC)"
]

def sort_by(items, base_path='.', method=None):
    def fullpath(x): return os.path.join(base_path, x)
    def get_timestamp(path):
        try: return os.path.getmtime(path)
        except FileNotFoundError: return float('-inf')

    if method == "Alphabetical (ASC)": return sorted(items)
    elif method == "Alphabetical (DESC)": return sorted(items, reverse=True)
    elif method == "Numerical (ASC)": return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]))
    elif method == "Numerical (DESC)": return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]), reverse=True)
    elif method == "Datetime (ASC)": return sorted(items, key=lambda x: get_timestamp(fullpath(x)))
    elif method == "Datetime (DESC)": return sorted(items, key=lambda x: get_timestamp(fullpath(x)), reverse=True)
    else: return items

class LoadVideoListFromDir:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "sort_method": (sort_methods, {"default": "Numerical (ASC)"}),
            },
            "optional": {
                "index_list": ("INT", {"default": 0, "tooltip": "Optional: Only load videos at these indices (after sorting).", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image_list", "file_paths")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    def run(self, directory, sort_method, index_list=None):
        import cv2
        # Handle list wrapping from INPUT_IS_LIST=True
        real_directory = directory[0] if isinstance(directory, list) else directory
        real_sort_method = sort_method[0] if isinstance(sort_method, list) else sort_method
        
        if not os.path.isdir(real_directory):
            raise FileNotFoundError(f"Directory '{real_directory}' not found.")
        
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        files = [f for f in os.listdir(real_directory) if os.path.splitext(f)[1].lower() in valid_extensions]
        
        if not files:
            return ([], [])

        # 1. Sort files
        files = sort_by(files, real_directory, real_sort_method)
        
        # 2. Filter by index_list if provided
        if index_list is not None:
            # index_list will be a list due to INPUT_IS_LIST=True
            indices = index_list
            filtered_files = []
            for i in indices:
                if 0 <= i < len(files):
                    filtered_files.append(files[i])
            files = filtered_files

        image_list = []
        path_list = []

        for f in files:
            path = os.path.join(real_directory, f)
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(torch.from_numpy(frame))
            cap.release()
            
            if frames:
                # Stack frames into [batch, h, w, c]
                video_tensor = torch.stack(frames, dim=0)
                image_list.append(video_tensor)
                path_list.append(path)
        
        print(f"[list-filter] Loaded {len(image_list)} videos from {real_directory}")
        return (image_list, path_list)

class GetVideoPathListFromDir:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "sort_method": (sort_methods, {"default": "Numerical (ASC)"}),
            },
            "optional": {
                "index_list": ("INT", {"default": 0, "tooltip": "Optional: Only get video paths at these indices (after sorting).", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_paths",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = "Gets video file paths from a directory without processing video frames."

    def run(self, directory, sort_method, index_list=None):
        real_directory = directory[0] if isinstance(directory, list) else directory
        real_sort_method = sort_method[0] if isinstance(sort_method, list) else sort_method

        if not os.path.isdir(real_directory):
            raise FileNotFoundError(f"Directory '{real_directory}' not found.")

        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        files = [f for f in os.listdir(real_directory) if os.path.splitext(f)[1].lower() in valid_extensions]

        if not files:
            return ([],)

        # 1. Sort files
        files = sort_by(files, real_directory, real_sort_method)

        # 2. Filter by index_list if provided
        if index_list is not None:
            indices = index_list
            filtered_files = []
            for i in indices:
                if 0 <= i < len(files):
                    filtered_files.append(files[i])
            files = filtered_files

        file_paths = [os.path.join(real_directory, f) for f in files]

        print(f"[list-filter] Found {len(file_paths)} video paths from {real_directory}")
        return (file_paths,)

class StringToIndex:
  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "string": ("STRING", {"default": "", "tooltip": "The string to be split into indices."}),
        "delimiter": ("STRING", {"default": ",", "tooltip": "The delimiter used to split the string."}),
      }
    }

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("index_list",)
  FUNCTION = "run"
  CATEGORY = NODE_CATEGORY
  INPUT_IS_LIST = False
  OUTPUT_IS_LIST = (True,)
  DESCRIPTION = "Splits a string into a list of indices based on the provided delimiter."

  def run(self, string, delimiter):
    if not isinstance(string, str):
        print(f"[list-filter] Warning: Expected string but got {type(string)}. Returning empty list.")
        return ([],)
    if not string.strip():
        return ([],)
    try:
        return ([int(i) for i in string.split(delimiter) if i.strip()],)
    except ValueError as e:
        print(f"[list-filter] Error: Could not convert part of string to integer: {e}")
        return ([],)

class FilterStringListByIndexList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_list": ("STRING", {"default": "", "tooltip": "The list of strings to be filtered."}),
                "index_list": ("INT", {"default": 0, "tooltip": "The list of indices to filter the string list."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_list",)
    OUTPUT_TOOLTIPS = ("The filtered list of strings.",)
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = "Filters the string list based on the provided index list."

    def run(self, string_list, index_list):
        return ([string_list[i] for i in index_list if i < len(string_list)],)

class FilterImageListByIndexList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE", {"tooltip": "The list of images to be filtered."}),
                "index_list": ("INT", {"default": 0, "tooltip": "The list of indices to filter the image list."}),
                "return_first_if_none": ("BOOLEAN", {"default": True, "tooltip": "Return the first image if the filtered list is empty."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_list",)
    OUTPUT_TOOLTIPS = ("The filtered list of images.",)
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = "Filters the image list based on the provided index list."

    def run(self, image_list, index_list, return_first_if_none):
        return_first_if_none_bool = return_first_if_none[0]
        filtered_list = [image_list[i] for i in index_list if i < len(image_list)]
        if not filtered_list and return_first_if_none_bool:
            filtered_list = [image_list[0]] if image_list else []
        return (filtered_list,)

class FilterAudioListByIndexList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("AUDIO", {"tooltip": "The list of audio to be filtered."}),
                "index_list": ("INT", {"default": 0, "tooltip": "The list of indices to filter the audio list."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio_list",)
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = "Filters the audio list based on the provided index list."

    def run(self, audio_list, index_list):
        return ([audio_list[i] for i in index_list if i < len(audio_list)],)

class FilterAnyListByIndexList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_list": ("*", {"tooltip": "Any list of data to be filtered."}),
                "index_list": ("INT", {"default": 0, "tooltip": "The list of indices to filter the list."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("filtered_list",)
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = "Filters ANY list of data based on the provided index list."

    def run(self, any_list, index_list):
        return ([any_list[i] for i in index_list if i < len(any_list)],)

class FindAnyStrings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_list": ("STRING", {"default": "", "tooltip": "The list of strings to search in."}),
                "search_strings": ("STRING", {"default": "", "tooltip": "The strings to search for."}),
                "delimiter": ("STRING", {"default": ",", "tooltip": "The delimiter used to split the search strings."}),
            }
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("found_index_list", "found_string_list",)
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    DESCRIPTION = "Checks if any of the search strings are present in the string list."

    def run(self, string_list, search_strings, delimiter):
        search_strings_str = search_strings[0]
        delimiter_str = delimiter[0]
        search_list = [s.strip() for s in search_strings_str.split(delimiter_str)]
        found_indices = [i for i, s in enumerate(string_list) if any(search in s for search in search_list)]
        found_strings = [string_list[i] for i in found_indices]
        return (found_indices, found_strings)

class FindNotAnyStrings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_list": ("STRING", {"default": "", "tooltip": "The list of strings to search in."}),
                "search_strings": ("STRING", {"default": "", "tooltip": "The list of strings to search for."}),
                "delimiter": ("STRING", {"default": ",", "tooltip": "The delimiter used to split the search strings."}),
            }
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("not_found_index_list", "not_found_string_list",)
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    DESCRIPTION = "Checks if none of the search strings are present in the string list."

    def run(self, string_list, search_strings, delimiter):
        search_strings_str = search_strings[0]
        delimiter_str = delimiter[0]
        search_list = [s.strip() for s in search_strings_str.split(delimiter_str)]
        not_found_indices = [i for i, s in enumerate(string_list) if all(search not in s for search in search_list)]
        not_found_strings = [string_list[i] for i in not_found_indices]
        return (not_found_indices, not_found_strings)

class RandomNormalDist:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mean": ("FLOAT", {"default": 0.0, "tooltip": "The mean of the normal distribution."}),
                "std_dev": ("FLOAT", {"default": 1.0, "tooltip": "The standard deviation of the normal distribution."}),
                "num_samples": ("INT", {"default": 10, "tooltip": "The number of random samples to generate."}),
                "min_value": ("FLOAT", {"default": 5.0, "tooltip": "The minimum value of the generated samples."}),
                "max_value": ("FLOAT", {"default": 8.0, "tooltip": "The maximum value of the generated samples."}),
            }
        }

    RETURN_TYPES = ("LIST","FLOAT")
    RETURN_NAMES = ("random_samples","first_sample")
    FUNCTION = "run"
    CATEGORY = "Random"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,False,)
    DESCRIPTION = "Generates random samples from a normal distribution."

    def run(self, mean, std_dev, num_samples, min_value, max_value):
        random_samples = []
        for _ in range(num_samples):
            sample = random.gauss(mean, std_dev)
            sample = max(min(sample, max_value), min_value)
            sample = round(sample, 1)
            random_samples.append(sample)
        return (random_samples,random_samples[0],)

NODE_CLASS_MAPPINGS = {
    "list_filter_LoadVideoListFromDir": LoadVideoListFromDir,
    "list_filter_GetVideoPathListFromDir": GetVideoPathListFromDir,
    "list_filter_StringToIndex": StringToIndex,
    "list_filter_FilterStringListByIndexList": FilterStringListByIndexList,
    "list_filter_FilterImageListByIndexList": FilterImageListByIndexList,
    "list_filter_FilterAudioListByIndexList": FilterAudioListByIndexList,
    "list_filter_FilterAnyListByIndexList": FilterAnyListByIndexList,
    "list_filter_FindAnyStrings": FindAnyStrings,
    "list_filter_FindNotAnyStrings": FindNotAnyStrings,
    "random_normal_dist": RandomNormalDist,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "list_filter_LoadVideoListFromDir": "Load Video List From Dir",
    "list_filter_GetVideoPathListFromDir": "Get Video Path List From Dir",
    "list_filter_StringToIndex": "Index List From String",
    "list_filter_FilterStringListByIndexList": "Filter String List",
    "list_filter_FilterImageListByIndexList": "Filter Image List",
    "list_filter_FilterAudioListByIndexList": "Filter Audio List",
    "list_filter_FilterAnyListByIndexList": "Filter Any List",
    "list_filter_FindAnyStrings": "Find Any Strings",
    "list_filter_FindNotAnyStrings": "Find Not Any Strings",
    "random_normal_dist": "Random Normal Distribution",
}
