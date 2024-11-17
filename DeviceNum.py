import subprocess


def get_device_index(device_name_substring):
    """
    Find the index of the device matching the given substring in its name.

    :param device_name_substring: A part of the device name to look for (case-insensitive).
    :return: The index of the matching device or None if not found.
    """
    try:
        # Run the 'arecord -l' command to list audio devices
        result = subprocess.run(['arecord', '-l'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"Error running 'arecord -l': {result.stderr}")
            return None

        # Search for the device index in the output
        for line in result.stdout.splitlines():
            if device_name_substring.lower() in line.lower():
                # Extract the device index (Card number)
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('card'):
                        return int(parts[i + 1].replace(':', ''))  # Get the number after "card"

        print(f"No device found matching '{device_name_substring}'")
        return None
    except Exception as e:
        print(f"Error detecting device index: {e}")
        return None


