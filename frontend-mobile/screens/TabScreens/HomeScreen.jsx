import React, { useRef, useState, useContext } from "react";
import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  Dimensions,
} from "react-native";
import { Avatar } from "react-native-paper";
import Ionicons from "react-native-vector-icons/Ionicons";
import Featured from "../../components/HomeScreenComp/Featured";
import Following from "../../components/HomeScreenComp/Following";
import { UserContext } from "../../context/UserContext";

const SCREEN_WIDTH = Dimensions.get("window").width;

export default function HomeScreen({ navigation }) {
  const { userData } = useContext(UserContext);
  const [activeScreenIndex, setActiveScreenIndex] = useState(0);
  const img1 = require("../../assets/img1.png");
  const scrollViewRef = useRef(null);

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
    <SafeAreaView style={{ backgroundColor: "#d9d9d9", flex: 1 }}>
      <View style={styles.uppertab}>
        <View style={styles.innertab}>
          <Avatar.Image
            source={
              userData.user.picture
                ? {
                    uri: `${url}/users/picture?path=${userData.user.picture}`,
                  }
                : img1
            }
            size={40}
            style={styles.headerimg}
          />
          <TouchableOpacity
            onPress={() => handleScreenPress(0)}
            style={{
              borderBottomColor: activeScreenIndex === 0 ? "#FF8216" : "white",
              borderBottomWidth: 3,
            }}
          >
            <Text
              style={{
                color: activeScreenIndex === 0 ? "#FF8216" : "gray",
                fontWeight: "bold",
              }}
            >
              Featured
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => handleScreenPress(1)}
            style={{
              borderBottomColor: activeScreenIndex === 1 ? "#FF8216" : "white",
              borderBottomWidth: 3,
            }}
          >
            <Text
              style={{
                color: activeScreenIndex === 1 ? "#FF8216" : "gray",
                fontWeight: "bold",
              }}
            >
              Following
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() =>
              navigation.navigate("Notification", {
                user: userData,
              })
            }
            disabled
          >
            <Ionicons name="notifications" color={"transparent"} size={30} />
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView
        ref={scrollViewRef}
        horizontal={true}
        pagingEnabled={true}
        showsHorizontalScrollIndicator={false}
        onMomentumScrollEnd={handleScroll}
      >
        <SafeAreaView style={styles.screen}>
          <Featured navigation={navigation} />
        </SafeAreaView>
        <SafeAreaView style={styles.screen}>
          <Following navigation={navigation} />
        </SafeAreaView>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  uppertab: {
    height: 70,
    backgroundColor: "white",
    marginBottom: 5,
    alignItems: "center",
  },
  innertab: {
    width: "95%",
    height: "100%",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  screen: { backgroundColor: "#d9d9d9", flex: 1, width: SCREEN_WIDTH },
  fullscreen: { width: "100%", height: "100%", flex: 1 },
});
