import React, { useState, useRef, useContext } from "react";
import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  Dimensions,
} from "react-native";

const SCREEN_WIDTH = Dimensions.get("window").width;

import Ionicons from "react-native-vector-icons/Ionicons";
import { Avatar } from "react-native-paper";
import * as ImagePicker from "react-native-image-picker/";
import { UserContext } from "../../context/UserContext";
import { AuthContext } from "../../auth/AuthContext";
import url from "../../utils/url";
import WalletModal from "../../components/ProfileHomeComp/WalletModal";
import FollowingModal from "../../components/ProfileHomeComp/FollowingModal";
import FollowerModal from "../../components/ProfileHomeComp/FollowerModal";
import MyProfileVideos from "../../components/ProfileHomeComp/MyProfileVideos";
import ColumnView from "../../components/ProfileHomeComp/ColumnView";
import img1 from "../../assets/img1.png";

export default function ProfileHomeScreen({ navigation }) {
  const [activeScreenIndex, setActiveScreenIndex] = useState(0);
  const [modalVisible, setModalVisible] = useState(false);
  const [followingmodalVisible, setFollowingModalVisible] = useState(false);
  const [followermodalVisible, setFollowerModalVisible] = useState(false);
  const [render, setRender] = useState(false);
  const { userData, getUser } = useContext(UserContext);
  const { user } = useContext(AuthContext);
  const scrollViewRef = useRef(null);

  const onCloseModal = (visible) => {
    setModalVisible(visible);
  };

  const onCloseModal1 = (visible) => {
    setFollowingModalVisible(visible);
  };
  const onCloseModal2 = (visible) => {
    setFollowerModalVisible(visible);
  };

  const selectImage = () => {
    ImagePicker.launchImageLibrary({}, async (response) => {
      if (response.didCancel) {
        console.log("User cancelled image picker");
      } else if (response.error) {
        console.log("ImagePicker Error: ", response.error);
      } else {
        console.log(response.assets[0].uri);
        await uploadPicture(response.assets[0].uri);
        await getUser();
        setRender((prev) => !prev);
      }
    });
  };

  const uploadPicture = async (uri) => {
    const formData = new FormData();
    formData.append("picture", {
      uri,
      type: "image/jpg",
      name: "picture",
    });
    try {
      const response = await fetch(`${url}/users/profilepicture`, {
        method: "patch",
        body: formData,
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${user}`,
        },
      });
      if (response.ok) {
        console.log("uploaded");
      } else {
        console.log("Something went wrong.");
      }
    } catch (error) {
      console.log(error);
    }
  };

  const handleScreenPress = (index) => {
    setActiveScreenIndex(index);
    scrollViewRef.current.scrollTo({
      x: index * SCREEN_WIDTH,
      y: 0,
      animated: true,
    });
  };

  const handleScroll = (event) => {
    const contentOffset = event.nativeEvent.contentOffset.x;
    const currentIndex = Math.round(contentOffset / SCREEN_WIDTH);
    setActiveScreenIndex(currentIndex);
  };

  return (
    <SafeAreaView style={{ backgroundColor: "white", flex: 1 }}>
      <View style={styles.uppertab}>
        <View style={styles.innertab}>
          <TouchableOpacity
            onPress={() =>
              navigation.navigate("MessageChat", { userData: userData })
            }
            style={{ width: "25%" }}
          >
            <Ionicons name="chatbubble" color={"#FF8216"} size={30} />
          </TouchableOpacity>
          <View>
            <Text
              style={{
                fontSize: 20,
                fontWeight: "medium",
                color: "black",
              }}
            >
              {userData.user.username}
            </Text>
          </View>
          <View
            style={{
              flexDirection: "row",
              width: "25%",
              justifyContent: "space-between",
            }}
          >
            <TouchableOpacity onPress={() => setModalVisible(true)}>
              <Ionicons name="wallet" color={"#FF8216"} size={30} />
            </TouchableOpacity>
            <TouchableOpacity onPress={() => navigation.navigate("Settings")}>
              <Ionicons
                name="ellipsis-horizontal"
                color={"#FF8216"}
                size={30}
              />
            </TouchableOpacity>
          </View>
        </View>
      </View>

      <ScrollView>
        <View style={styles.imageicon}>
          <Avatar.Image
            source={
              userData.user.picture
                ? {
                    uri: `${url}/users/picture?path=${userData.user.picture}`,
                  }
                : img1
            }
            size={110}
          />
          <TouchableOpacity
            style={{ marginLeft: 100, marginBottom: 10 }}
            onPress={selectImage}
          >
            <Ionicons
              name="camera-outline"
              size={25}
              style={{
                backgroundColor: "darkorange",
                color: "white",
                borderRadius: 20,
                padding: 2,
              }}
            />
          </TouchableOpacity>
          <Text
            style={{
              fontSize: 18,
              fontWeight: "semibold",
              color: "darkgray",
            }}
          >
            @{userData.user.username}
          </Text>
        </View>

        <View style={styles.followercount}>
          <TouchableOpacity
            style={{ alignItems: "center", width: "30%" }}
            onPress={() => setFollowingModalVisible(true)}
          >
            <Text style={styles.followtext}>
              {userData.user.following.length || 0}
            </Text>
            <Text style={{ color: "darkgray" }}>Following</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={{ alignItems: "center", width: "30%" }}
            onPress={() => setFollowerModalVisible(true)}
          >
            <Text style={styles.followtext}>
              {userData.user.followers.length || 0}
            </Text>
            <Text style={{ color: "darkgray" }}>Followers</Text>
          </TouchableOpacity>

          <View style={{ alignItems: "center", width: "30%" }}>
            <Text style={styles.followtext}>{userData.user.sales}</Text>
            <Text style={{ color: "darkgray" }}>Sales</Text>
          </View>
        </View>

        <View style={styles.followercount1}>
          <TouchableOpacity
            onPress={() => handleScreenPress(0)}
            style={{
              width: "30%",
              borderBottomColor: activeScreenIndex === 0 ? "#FF8216" : "white",
              borderBottomWidth: 3,
              alignItems: "center",
            }}
          >
            <Ionicons
              name={activeScreenIndex === 0 ? "list" : "list-outline"}
              color={activeScreenIndex === 0 ? "#FF8216" : "gray"}
              size={25}
            />
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => handleScreenPress(1)}
            style={{
              width: "30%",
              borderBottomColor: activeScreenIndex === 1 ? "#FF8216" : "white",
              borderBottomWidth: 3,
              alignItems: "center",
            }}
          >
            <Ionicons
              name={activeScreenIndex === 1 ? "grid" : "grid-outline"}
              color={activeScreenIndex === 1 ? "#FF8216" : "gray"}
              size={25}
            />
          </TouchableOpacity>
        </View>

        <ScrollView
          ref={scrollViewRef}
          horizontal={true}
          pagingEnabled={true}
          showsHorizontalScrollIndicator={false}
          onMomentumScrollEnd={handleScroll}
        >
          <SafeAreaView style={styles.screen}>
            <MyProfileVideos render={render} />
          </SafeAreaView>
          <SafeAreaView style={styles.screen1}>
            <ColumnView />
          </SafeAreaView>
        </ScrollView>
      </ScrollView>
      <WalletModal
        modalVisible={modalVisible}
        onCloseModal={onCloseModal}
        user={userData.user}
      />
      <FollowingModal
        modalVisible={followingmodalVisible}
        onCloseModal={onCloseModal1}
        user={userData.user}
      />
      <FollowerModal
        modalVisible={followermodalVisible}
        onCloseModal={onCloseModal2}
        user={userData.user}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  uppertab: {
    height: 70,
    backgroundColor: "white",
    marginBottom: 5,
    alignItems: "center",
    shadowColor: "#171717",
    shadowOffset: { width: -2, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 10,
  },
  innertab: {
    width: "92%",
    height: "100%",
    flexDirection: "row",
    alignItems: "center",
    marginTop: "auto",
    justifyContent: "space-between",
  },
  imageicon: {
    alignItems: "center",
    justifyContent: "space-around",
    height: 150,
    marginTop: 10,
  },
  followercount: {
    flexDirection: "row",
    height: 50,
    width: "85%",
    alignSelf: "center",
    justifyContent: "space-around",
    marginTop: 15,
    marginBottom: 15,
  },
  followtext: { fontSize: 20, fontWeight: "semibold", color: "black" },
  followercount1: {
    flexDirection: "row",
    height: 45,
    width: "85%",
    alignSelf: "center",
    justifyContent: "space-around",
    marginTop: 10,
    marginBottom: 10,
  },
  screen: { backgroundColor: "#d9d9d9", flex: 1, width: SCREEN_WIDTH },
  screen1: { backgroundColor: "white", flex: 1, width: SCREEN_WIDTH },
});
