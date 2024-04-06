import { useState, useEffect, useRef, useContext } from "react";
import Video from "react-native-video";
import {
  View,
  StyleSheet,
  Dimensions,
  TouchableOpacity,
  StatusBar,
} from "react-native";
import Orientation from "react-native-orientation-locker";
import ProgressBar from "./ProgressBar";
import PlayerControls from "./PlayerControls";
import { FullscreenClose, FullscreenOpen } from "../../assets/svgs";
import url from "../../utils/url";

const width = Dimensions.get("screen").width;
const height = Dimensions.get("screen").height;

const VideoPlayer = ({ source, fullscreenIcon, recorded }) => {
  const videoRef = useRef();
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [play, setPlay] = useState(false);
  const [showControl, setShowControl] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    Orientation.addOrientationListener(handleOrientation);
    return () => {
      Orientation.removeOrientationListener(handleOrientation);
    };
  }, []);

  const handleOrientation = (orientation) => {
    if (orientation === "LANDSCAPE-LEFT" || orientation === "LANDSCAPE-RIGHT") {
      setFullscreen(true);
      StatusBar.setHidden(true);
    } else {
      setFullscreen(false);
      StatusBar.setHidden(false);
    }
  };

  const handlePlayPause = () => {
    if (play) {
      setPlay(false);
      setShowControl(true);
      return;
    }
    setTimeout(() => setShowControl(false), 2000);
    setPlay(true);
  };

  const handlePlay = () => {
    setTimeout(() => setShowControl(false), 500);
    setPlay(true);
  };

  const skipBackward = () => {
    videoRef.current.seek(currentTime - 15);
    setCurrentTime(currentTime - 15);
  };

  const skipForward = () => {
    videoRef.current.seek(currentTime + 15);
    setCurrentTime(currentTime + 15);
  };

  const handleControls = () => {
    if (showControl) {
      setShowControl(false);
    } else {
      setShowControl(true);
    }
  };

  const handleFullscreen = () => {
    if (fullscreen) {
      Orientation.unlockAllOrientations();
    } else {
      Orientation.lockToLandscapeLeft();
    }
  };

  const onLoadEnd = (data) => {
    setDuration(data.duration);
    setCurrentTime(data.currentTime);
  };

  const onProgress = (data) => {
    setCurrentTime(data.currentTime);
  };

  const onSeek = (data) => {
    videoRef.current.seek(data.seekTime);
    setCurrentTime(data.seekTime);
  };

  const onEnd = () => {
    setPlay(false);
    videoRef.current.seek(0);
  };

  return (
    <View style={fullscreen ? styles.fullscreenContainer : styles.container}>
      <TouchableOpacity onPress={handleControls}>
        <>
          <Video
            ref={videoRef}
            source={
              recorded
                ? { uri: source }
                : {
                    uri: `${url}/videos/stream?path=${source}`,
                    type: "video/mp4",
                    headers: {
                      range: "byte=0-",
                    },
                  }
            }
            style={fullscreen ? styles.fullscreenVideo : styles.video}
            controls={false}
            // resizeMode={
            // 	fullscreenIcon
            // 		? fullscreen
            // 			? 'contain'
            // 			: 'stretch'
            // 		: 'contain'
            // }
            onLoad={onLoadEnd}
            onProgress={onProgress}
            onEnd={onEnd}
            paused={!play}
            volume={1}
          />
          {showControl && (
            <View style={styles.controlOverlay}>
              {fullscreenIcon && (
                <TouchableOpacity
                  onPress={handleFullscreen}
                  hitSlop={{
                    top: 10,
                    bottom: 10,
                    left: 10,
                    right: 10,
                  }}
                  style={styles.fullscreenButton}
                >
                  {fullscreen ? <FullscreenClose /> : <FullscreenOpen />}
                </TouchableOpacity>
              )}
              <PlayerControls
                onPlay={handlePlay}
                onPause={handlePlayPause}
                playing={play}
                skipBackwards={skipBackward}
                skipForwards={skipForward}
              />

              <ProgressBar
                currentTime={currentTime}
                duration={duration > 0 ? duration : 0}
                onSlideStart={handlePlayPause}
                onSlideComplete={handlePlayPause}
                onSlideCapture={onSeek}
              />
            </View>
          )}
        </>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 0,
    flex: 1,
  },
  fullscreenContainer: {
    flex: 1,
    backgroundColor: "#ebebeb",
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    zIndex: 5,
  },
  video: {
    height: "100%",
    width: "100%",
    backgroundColor: "black",
  },
  fullscreenVideo: {
    flex: 1,
    height: height,
    width: width,
    backgroundColor: "black",
  },
  text: {
    marginTop: 30,
    marginHorizontal: 20,
    fontSize: 15,
    textAlign: "justify",
  },
  fullscreenButton: {
    flex: 1,
    flexDirection: "row",
    alignSelf: "flex-end",
    alignItems: "center",
    paddingRight: 10,
  },
  controlOverlay: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "#000000c4",
    justifyContent: "space-between",
  },
});

export default VideoPlayer;
