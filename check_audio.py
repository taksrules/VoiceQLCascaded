import pyaudio

def get_default_output_device():
    p = pyaudio.PyAudio()
    try:
        default_info = p.get_default_output_device_info()
        print("\n=== Default Output Device ===")
        print(f"Index: {default_info['index']}")
        print(f"Name: {default_info['name']}")
        print(f"Max Channels: {default_info['maxOutputChannels']}")
        print(f"Default Sample Rate: {default_info['defaultSampleRate']}")
        print("==============================\n")
    except Exception as e:
        print(f"Error getting default output device: {e}")
    
    print("=== All Output Devices ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"[{i}] {info['name']} (Channels: {info['maxOutputChannels']}, Rate: {info['defaultSampleRate']})")
    
    p.terminate()

if __name__ == "__main__":
    get_default_output_device()
