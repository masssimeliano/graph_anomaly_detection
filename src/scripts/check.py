import torch


def main():
    device = torch.device("cuda:0")

    # Всего памяти
    total_memory = torch.cuda.get_device_properties(device).total_memory
    # Занятая память
    allocated_memory = torch.cuda.memory_allocated(device)
    # Зарезервированная память (включает кэш)
    reserved_memory = torch.cuda.memory_reserved(device)

    print(f"Total memory: {total_memory / 1024 ** 3:.2f} GB")
    print(f"Allocated memory: {allocated_memory / 1024 ** 3:.2f} GB")
    print(f"Reserved memory: {reserved_memory / 1024 ** 3:.2f} GB")
    print(f"Free (reserved - allocated): {(reserved_memory - allocated_memory) / 1024 ** 3:.2f} GB")

if __name__ == "__main__":
    main()