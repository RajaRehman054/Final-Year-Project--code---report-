import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
  useContext,
} from "react";
import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  Image,
} from "react-native";
import { Camera, useCameraDevices } from "react-native-vision-camera";
import Feather from "react-native-vector-icons/Feather";
import Ionicons from "react-native-vector-icons/Ionicons";
import { Stopwatch } from "react-native-stopwatch-timer";
import { FFmpegKit } from "ffmpeg-kit-react-native";
import fs from "react-native-fs";
import { useIsFocused } from "@react-navigation/native";
import Reanimated, {
  useAnimatedProps,
  useSharedValue,
  withSpring,
} from "react-native-reanimated";
import { useFocusEffect } from "@react-navigation/native";
import Orientation from "react-native-orientation-locker";
import { UserContext } from "../../context/UserContext";
import Loader from "../../components/Loader";
import Geolocation from '@react-native-community/geolocation';
import Geocoder from 'react-native-geocoding';
Geocoder.init("AIzaSyCIAQSWoAC-7bSHWyUEcUdUGwUfiEWqF3U"); 

const path = fs.DocumentDirectoryPath + "/test.png";
fs.readFileAssets("images/image.png", "base64")
  .then((binary) => {
    fs.writeFile(path, binary, "base64")
      .then((success) => {
        console.log("FILE WRITTEN!");
      })
      .catch(console.error);
  })
  .catch(console.error);

const ReanimatedCamera = Reanimated.createAnimatedComponent(Camera);
Reanimated.addWhitelistedNativeProps({
  zoom: true,
});

export default function CameraRecording({ navigation, route }) {
  const { userData } = useContext(UserContext);
  const [location, setLocation] = useState({ city: 'Islamabad', country: 'Pakistan' });
  const [displayText, setDisplayText] = useState("");
  const [recording, setRecording] = useState(false);
  const [direction, setDirection] = useState("back");
  const [check, setCheck] = useState(true);
  const [isStopwatchStart, setIsStopwatchStart] = useState(false);
  const [loading, setLoading] = useState(false);
  const cameraRef = useRef(null);
  const devices = useCameraDevices();
  const selectedDevice = devices.back;
  const isScreenFocused = useIsFocused();
  const zoom = useSharedValue(0);
  const img1 = require("../../assets/penguin.gif");
  const [hirer, setHirer] = useState([]);
  const [isRecordingInProgress, setIsRecordingInProgress] = useState(false);
  
  const [address, setAddress] = useState("");

  const getCurrentLocation = () => {
    // Get current position
    Geolocation.getCurrentPosition(
      position => {
        const { latitude, longitude } = position.coords;
        Geocoder.from(latitude, longitude)
          .then(json => {
            const formattedAddress = json.results[0].formatted_address;
            if (formattedAddress) {
              const addressParts = formattedAddress.split(', ');
              const extractedCity = addressParts[2];
              const extractedCountry = addressParts[addressParts.length - 1];
              setAddress(formattedAddress);
              setLocation({ city: extractedCity, country: extractedCountry });
            }
          })
          .catch(error => console.warn(error));
      },
      error => console.warn(error),
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000 }
    );
  };

  useFocusEffect(
    useCallback(() => {
      Orientation.lockToLandscape();
      return () => {
        Orientation.lockToPortrait();
        setLoading(false);
      };
    }, [])
  );
  useEffect(() => {
    getCurrentLocation();
  }, []);

  useEffect(() => {
    if (userData.user.hirer !== null) {
      const getHirer = async () => {
        try {
          const res = await fetch(
            `${url}/users?userId=${userData.user.hirer}`,
            {
              headers: {
                "content-type": "application/json",
              },
            }
          );
          const response = await res.json();
          setHirer(response);
        } catch (err) {
          console.log(err);
        }
      };
      getHirer();
    }
  }, [userData.user]);

  useEffect(() => {
    const getPermission = async () => {
      const permission = await Camera.requestCameraPermission();
      const newMicrophonePermission =
        await Camera.requestMicrophonePermission();
      if (permission === "denied") await Linking.openSetting();
      if (newMicrophonePermission === "denied") await Linking.openSetting();
    };
    if (route.params && route.params.text) {
      setDisplayText(route.params.text);
    }
    getPermission();
  }, [route.params]);

  const zoomIncrease = useCallback(() => {
    zoom.value = withSpring(zoom.value + 1);
  }, []);

  const zoomDecrease = useCallback(() => {
    zoom.value = withSpring(zoom.value - 1);
  }, []);

  const animatedProps = useAnimatedProps(() => ({ zoom: zoom.value }), [zoom]);

  const start = async () => {
    setIsRecordingInProgress(true);
    const watermarkText =
      userData.user.hirer === null ? userData.user.username : hirer.username;
    const watermarkText2 = `${location.city} ${location.country}`;
    setRecording(true);
    setIsStopwatchStart(true);
    await cameraRef.current.startRecording({
      flash: "off",
      onRecordingFinished: async (video) => {
        Orientation.lockToPortrait();
        setLoading(true);
        setRecording(false);
        const output = video.path.split(".mp4") + "con.mp4";
        const command = `-i ${video.path} -i ${path} -filter_complex "[0:v][1:v] overlay=x=(main_w-overlay_w):0,drawtext=fontfile=/system/fonts/DroidSans.ttf:text='${watermarkText}':x=w-tw-50:y=100:fontsize=25:fontcolor=orange,drawtext=fontfile=/system/fonts/DroidSans.ttf:text='${watermarkText2}':x=w-tw-50:y=h-th-50:fontsize=25:fontcolor=orange" -codec:v libx264 -profile:v high -level:v 4.2 -preset ultrafast -crf 18 -r 30 -movflags +faststart -c:a aac ${output}`;
        await FFmpegKit.execute(command);
        navigation.navigate("PostContentScreen", { uri: output , address:address});
      },
      onRecordingError: (error) => console.error(error),
    });
  };
  

  const stop = async () => {
    await cameraRef.current.stopRecording();
  };

  const changeCamera = async () => {
    if (direction === "back") setDirection("front");
    else setDirection("back");
  };

  const pause = async () => {
    if (check) {
      await cameraRef.current.pauseRecording();
      setCheck(false);
      setIsStopwatchStart(false);
    } else {
      await cameraRef.current.resumeRecording();
      setCheck(true);
      setIsStopwatchStart(true);
    }
  };

  const clearText = () => {
    setDisplayText("");
  };

  if (selectedDevice == null || loading) {
    return (
      <>
        <View style={styles.container1}>
          <View style={{ alignItems: "center", marginTop: 100 }}>
            <Text style={{ color: "black", marginTop: 40 }}>
              Processing Video...
            </Text>
            <Image source={img1} style={styles.image} />
          </View>
          <Loader />
        </View>
      </>
    );
  }

  const ScriptView = () => {
    return (
      <View style={{ height: "100%" }}>
        <View style={{ height: "15%" }}>
          <TouchableOpacity
            style={{
              width: "15%",
              marginLeft: "auto",
              alignItems: "center",
            }}
            onPress={() => clearText()}
          >
            <Ionicons name="close-circle" color={"white"} size={30} />
          </TouchableOpacity>
        </View>
        <ScrollView style={{ height: "85%" }}>
          <Text
            style={{
              fontSize: 50,
              color: "white",
              fontWeight: "bold",
              textAlign: "center",
            }}
          >
            {displayText}
          </Text>
        </ScrollView>
      </View>
    );
  };
  return (
    <SafeAreaView style={{ flex: 1 }}>
      <ReanimatedCamera
        ref={cameraRef}
        device={direction === "front" ? devices.front : devices.back}
        isActive={isScreenFocused}
        animatedProps={animatedProps}
        enableZoomGesture={true}
        style={StyleSheet.absoluteFill}
        video={true}
        audio={true}
        photo={true}
        enableHighQualityPhotos={true}
      />
      <View style={styles.uppertab}>
        <View style={styles.innertab}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Ionicons name="chevron-back" color={"white"} size={40} />
          </TouchableOpacity>
        </View>
      </View>
      <View style={styles.middletab}>
        <View style={{ width: "16%" }}></View>
        <View style={{ width: "68%", height: "100%" }}>
          {displayText === "" ? <View></View> : <ScriptView />}
        </View>
        {isRecordingInProgress ? (
          <View style={{ width: "16%" }}></View>
        ) : (
          <View style={{ width: "16%" }}>
            <TouchableOpacity style={{ margin: 10 }} onPress={changeCamera}>
              <Ionicons name="sync" color={"white"} size={30} />
            </TouchableOpacity>
            <TouchableOpacity style={{ margin: 10 }}>
              <Ionicons name="flash" color={"white"} size={30} />
            </TouchableOpacity>
            <TouchableOpacity style={{ margin: 10 }}>
              <Feather
                name="zoom-in"
                color="white"
                size={30}
                onPress={zoomIncrease}
              />
            </TouchableOpacity>
            <TouchableOpacity style={{ margin: 10 }}>
              <Feather
                name="zoom-out"
                color="white"
                size={30}
                onPress={zoomDecrease}
              />
            </TouchableOpacity>
          </View>
        )}
      </View>

      <View style={styles.lasttab}>
        {!recording && (
          <>
            <TouchableOpacity
              style={[
                styles.lasttabicon,
                {
                  height: 70,
                  width: 70,
                  borderWidth: 5,
                  backgroundColor: "green",
                },
              ]}
              onPress={() => navigation.navigate("Teleprompter")}
            >
              <Ionicons name="receipt" color={"white"} size={40} />
            </TouchableOpacity>
            <TouchableOpacity
              onPress={start}
              style={[
                styles.lasttabicon,
                {
                  height: 90,
                  width: 90,
                  borderWidth: 8,
                  backgroundColor: "red",
                },
              ]}
            >
              <Ionicons name="videocam" color={"white"} size={40} />
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                {
                  height: 70,
                  width: 70,
                  backgroundColor: "transparent",
                },
              ]}
              disabled
            >
              <Ionicons name="albums" color={"transparent"} size={40} />
            </TouchableOpacity>
          </>
        )}
        {recording && (
          <>
            <TouchableOpacity
              onPress={stop}
              style={[
                styles.lasttabicon,
                {
                  height: 70,
                  width: 70,
                  borderWidth: 5,
                },
              ]}
            >
              <Ionicons name="stop" color={"red"} size={40} />
            </TouchableOpacity>
            <TouchableOpacity
              onPress={pause}
              style={[
                styles.lasttabicon,
                {
                  height: 90,
                  width: 90,
                  borderWidth: 8,
                },
              ]}
            >
              {check ? (
                <Ionicons name="pause" color={"red"} size={40} />
              ) : (
                <Ionicons name="play" color={"red"} size={40} />
              )}
            </TouchableOpacity>
            <Stopwatch
              laps={true}
              secs
              start={isStopwatchStart}
              options={options}
            />
          </>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  uppertab: {
    height: "10%",
    alignItems: "center",
  },
  innertab: {
    width: "92%",
    height: "100%",
    flexDirection: "row",
    alignItems: "center",
    marginTop: "auto",
  },
  middletab: {
    height: "60%",
    alignItems: "center",
    justifyContent: "space-between",
    flexDirection: "row",
  },
  lasttab: {
    height: "30%",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },
  lasttabicon: {
    margin: 10,
    alignItems: "center",
    borderColor: "white",
    justifyContent: "center",
    borderRadius: 50,
  },
  container1: {
    flex: 1,

    alignItems: "center",
    backgroundColor: "white",
  },
  image: {
    width: 200,
    height: 200,
  },
});

const options = {
  container: {
    margin: 10,
    alignItems: "center",
    justifyContent: "center",
    height: 70,
    width: 70,
  },
  text: {
    fontSize: 18,
    color: "red",
  },
};
