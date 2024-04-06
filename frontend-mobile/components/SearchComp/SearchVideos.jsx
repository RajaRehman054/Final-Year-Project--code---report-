import React, { useState } from "react";
import { Text, View, TouchableOpacity, StyleSheet } from "react-native";
import { Avatar } from "react-native-paper";
import Ionicons from "react-native-vector-icons/Ionicons";
import url from "../../utils/url";
import VideoPlayer from "../VideoPlayerComp/VideoPlayer";
import img1 from "../../assets/img1.png";

export default function SearchVideos({ video, navigation }) {
  const [heartFilled, setHeartFilled] = useState(false);
  const onPressHeart = () => {
    setHeartFilled(!heartFilled);
  };

  return (
    <>
      {video.map((e, index) => (
        <View key={index} style={styles.contentbar}>
          <View style={styles.profileandstateview}>
            <View style={styles.profilebar}>
              <Avatar.Image
                source={
                  e.owner.picture
                    ? {
                        uri: `${url}/users/picture?path=${e.owner.picture}`,
                      }
                    : img1
                }
                size={45}
              />
              <View style={{ marginLeft: 10 }}>
                <TouchableOpacity
                  onPress={() =>
                    navigation.navigate("OthersProfile", {
                      user1: e,
                    })
                  }
                >
                  <Text
                    style={{
                      fontSize: 15,
                      color: "black",
                      fontWeight: "bold",
                    }}
                  >
                    {e.owner.username}
                  </Text>
                </TouchableOpacity>
                <Text style={{ fontSize: 10, color: "gray" }}>
                  {e.created_on}
                </Text>
              </View>
            </View>
            <TouchableOpacity disabled>
              <Ionicons
                name="ellipsis-horizontal"
                color={"transparent"}
                size={30}
              />
            </TouchableOpacity>
          </View>

          <View style={styles.tagview}>
            <Text style={{ color: "black" }}>{e.description}</Text>
            <Text style={{ color: "#FF8216" }}>{e.tags}</Text>
          </View>

          <View style={styles.videoview}>
            <VideoPlayer fullscreenIcon={false} source={e.path} />
          </View>

          <View style={styles.profileandstateview}>
            <TouchableOpacity style={styles.icon} onPress={onPressHeart}>
              <Ionicons
                name={heartFilled ? "heart" : "heart-outline"}
                color={"#FF8216"}
                size={30}
              />
              <Text style={{ color: "#FF8216" }}>16</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.icon}>
              <Ionicons name="chatbox-outline" color={"gray"} size={25} />
              <Text style={{ color: "gray" }}>16</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.icon}
              onPress={() =>
                navigation.navigate("BuyContent", {
                  video: e,
                })
              }
            >
              <Ionicons name="download-outline" color={"gray"} size={25} />
              <Text style={{ color: "gray" }}>16</Text>
            </TouchableOpacity>
          </View>
        </View>
      ))}
    </>
  );
}
const styles = StyleSheet.create({
  backgroundVideo: {
    height: "50%",
    width: "92%",
  },
  contentbar: {
    backgroundColor: "white",
    height: 350,
    alignItems: "center",
    borderBottomColor: "#d9d9d9",
    borderBottomWidth: 8,
  },
  profileandstateview: {
    width: "92%",
    height: "15%",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  profilebar: {
    flexDirection: "row",
    width: "90%",
    marginTop: "auto",
    alignItems: "center",
  },
  tagview: {
    width: "92%",
    height: "15%",
    justifyContent: "space-around",
  },
  videoview: {
    height: "55%",
    width: "92%",
    borderRadius: 20,
  },
  imgs: {
    width: "100%",
    height: "100%",
    borderRadius: 20,
  },
  icon: {
    width: "25%",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-around",
  },
});
