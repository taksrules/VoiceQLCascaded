import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    print("\n" + "="*60)
    print(f"{'INDEX':<7} {'DEVICE NAME':<40} {'OUT CH':<7} {'RATE'}")
    print("="*60)

    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        name = device_info.get('name')
        out_channels = device_info.get('maxOutputChannels')
        rate = int(device_info.get('defaultSampleRate'))
        
        # Only show devices with output channels
        if out_channels > 0:
            print(f"{i:<7} {name[:38]:<40} {out_channels:<7} {rate}")

    print("="*60)
    print("\nPRO TIP: Look for 'Speaker' or your Headphone name.")
    print("If you see multiple, try the one that matches your current Windows output.")
    p.terminate()

if __name__ == "__main__":
    list_audio_devices()
