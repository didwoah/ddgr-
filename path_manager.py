import os
import json

class PathManager():
    def __init__(self, args, base_path="./save", map_path = None):
        self.base_path = base_path
        self.__root_path = self.create_root_path()
        self.__now_task = 0
        self.__map_path = map_path
        self._save_args_to_file(args)

    def get_map(self):
        if self.__map_path is None:
            self.__map_path = os.path.join(self.__root_path, 'map.json')
            with open(self.__map_path, "w") as map_file:
                json.dump({}, map_file)
        with open(self.__map_path, "r") as map_file:
            map = json.load(map_file)

        int_key_map = {int(key): value for key, value in map.items()}
        
        return int_key_map
    
    def set_map(self, map):
        with open(self.__map_path, "w") as map_file:
            json.dump(map, map_file)
        return map
    
    def _save_args_to_file(self, args, file_path="args.txt"):

        file_path = os.path.join(self.__root_path, file_path)
        with open(file_path, "w") as file:
            json.dump(vars(args), file, indent=4)

    def create_root_path(self):
        """
        Create a root path based on the argparse arguments.
        """
        os.makedirs(self.base_path, exist_ok=True)

        existing_folders = [folder for folder in os.listdir(self.base_path) if folder.startswith("experiments")]
        existing_numbers = []

        for folder in existing_folders:
            try:
                number = int(folder.replace("experiments", ""))
                existing_numbers.append(number)
            except ValueError:
                pass

        next_number = max(existing_numbers) + 1 if existing_numbers else 0
        root_path = os.path.join(self.base_path, f"experiments{next_number}")

        os.makedirs(root_path, exist_ok=True)
        print(f"Root path created: {root_path}")

        return root_path
    
    def _get_task_root_path(self):
        task_root_path = os.path.join(self.__root_path, f'task{self.__now_task}')
        os.makedirs(task_root_path, exist_ok = True)

        return task_root_path
        
    def get_model_path(self, model):
        assert model in ['classifier', 'generator']

        task_root_path = self._get_task_root_path()
        model_dir_path = os.path.join(task_root_path, model)

        os.makedirs(model_dir_path, exist_ok = True)

        model_path = os.path.join(model_dir_path, 'weights.pth')
        
        return model_path
    
    def get_image_path(self, aug = False):

        task_root_path = self._get_task_root_path()
        if aug:
            image_path = os.path.join(task_root_path, 'augmented_image')
        else:
            image_path = os.path.join(task_root_path, 'generated_image')

        os.makedirs(image_path, exist_ok = True)

        return image_path
    
    def get_results_path(self):

        task_root_path = self._get_task_root_path()
        results_path = os.path.join(task_root_path, 'results')

        os.makedirs(results_path, exist_ok = True)

        return results_path

    def _get_prev_task_root_path(self):
        assert self.__now_task > 0
        task_root_path = os.path.join(self.__root_path, f'task{self.__now_task - 1}')
        os.makedirs(task_root_path, exist_ok = True)

        return task_root_path

    def get_prev_model_path(self, model):
        assert model in ['classifier', 'generator']

        task_root_path = self._get_prev_task_root_path()
        model_path = os.path.join(task_root_path, model, 'weights.pth')

        return model_path

    def update_task_count(self):
        self.__now_task += 1