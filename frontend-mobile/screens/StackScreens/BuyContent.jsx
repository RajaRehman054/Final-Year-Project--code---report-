import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from "react-native";

import Ionicons from "react-native-vector-icons/Ionicons";
import { Avatar } from "react-native-paper";
import img1 from "../../assets/img1.png";
import url from "../../utils/url";
import VideoPlayer from "../../components/VideoPlayerComp/VideoPlayer";

export default function BuyContent({ navigation, route }) {
  const { video } = route.params;
  return (
    <SafeAreaView style={{ backgroundColor: "white", flex: 1 }}>
      <View style={styles.uppertab}>
        <View style={styles.innertab}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Ionicons name="chevron-back" color={"white"} size={30} />
          </TouchableOpacity>
          <View>
            <Text
              style={{
                fontSize: 20,
                fontWeight: "bold",
                color: "white",
              }}
            >
              Buy Content
            </Text>
          </View>
          <View>
            <Ionicons name="ellipsis-horizontal" color={"#FF8216"} size={30} />
          </View>
        </View>
      </View>
      <ScrollView>
        <View
          style={{
            backgroundColor: "#FF8216",
            borderBottomLeftRadius: 20,
            borderBottomRightRadius: 20,
          }}
        >
          <View
            style={{
              borderColor: "white",
              borderWidth: 1,
              width: "92%",
              height: 200,
              alignSelf: "center",
              marginTop: 5,
            }}
          >
            <VideoPlayer fullscreenIcon={false} source={video.path} />
          </View>
          <View style={styles.infotext}>
            <View style={styles.iconimg}>
              <Avatar.Image
                source={
                  video.owner.picture
                    ? {
                        uri: `${url}/users/picture?path=${video.owner.picture}`,
                      }
                    : img1
                }
                size={50}
              />
            </View>
            <View style={styles.notifytext}>
              <View>
                <Text style={{ color: "white" }}>
                  <Text
                    style={{
                      fontWeight: "bold",
                      color: "white",
                      fontSize: 16,
                    }}
                  >
                    {video.owner.username}
                  </Text>
                  {"   "}@{video.owner.username}
                </Text>

                <Text style={{ color: "white" }}>{video.created_on}</Text>
              </View>
            </View>
          </View>
        </View>
        <View
          style={{
            width: "92%",
            alignSelf: "center",
            marginTop: 30,
          }}
        >
          <TouchableOpacity
            style={{
              height: 50,
              backgroundColor: "#FF8216",
              borderRadius: 10,
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            <Text
              style={{
                fontWeight: "bold",
                color: "white",
                fontSize: 20,
              }}
            >
              Buy 20$
            </Text>
          </TouchableOpacity>

          <View>
            <Text
              style={{
                color: "black",
                justifyContent: "center",
                alignItems: "center",
                marginVertical: 10,
              }}
            >
              {video.description}
            </Text>
            <Text
              style={{
                color: "black",
                justifyContent: "center",
                alignItems: "center",
                marginVertical: 10,
              }}
            >
              {video.tags}
            </Text>
          </View>
        </View>
        <View
          style={{
            backgroundColor: "#d9d9d9",
            height: 34,
            justifyContent: "center",
          }}
        >
          <View style={{ width: "92%", alignSelf: "center" }}>
            <Text
              style={{
                color: "darkgray",
                fontWeight: "bold",
                fontSize: 18,
              }}
            >
              Content Details
            </Text>
          </View>
        </View>
        <View style={{ width: "92%", alignSelf: "center" }}>
          <View style={{ marginTop: 10 }}>
            <Text style={{ color: "black" }}>Captured on:</Text>
          </View>
          <View style={{ marginTop: 10 }}>
            <Text style={{ color: "black" }}>City</Text>
          </View>
          <View style={{ marginTop: 10 }}>
            <Text style={{ color: "black" }}>Country</Text>
          </View>
          <View style={{ marginTop: 10 }}>
            <Text style={{ color: "black" }}>Category</Text>
          </View>
          <View style={{ marginTop: 10, marginBottom: 50 }}>
            <Text style={{ color: "black" }}>Video Ratio</Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  uppertab: {
    height: 70,
    backgroundColor: "#FF8216",
    alignItems: "center",
  },
  innertab: {
    width: "92%",
    height: "100%",
    flexDirection: "row",
    alignItems: "center",
    marginTop: "auto",
    justifyContent: "space-between",
  },

  iconimg: {
    width: "25%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  notifytext: {
    width: "75%",
    height: "100%",
    justifyContent: "center",
  },
  infotext: {
    width: "92%",
    height: 80,
    alignSelf: "center",
    backgroundColor: "#FF8216",
    flexDirection: "row",
  },
});
