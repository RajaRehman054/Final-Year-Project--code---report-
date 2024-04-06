import React, { useState, useContext, useEffect} from "react";
import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  TextInput,
  ScrollView,
} from "react-native";
import Ionicons from "react-native-vector-icons/Ionicons";

import VideoPlayer from "../../components/VideoPlayerComp/VideoPlayer";
import { AuthContext } from "../../auth/AuthContext";
import Loader from "../../components/Loader";
import Toast from "react-native-toast-message";


export default function PostContentScreen({ navigation, route }) {
  const { address } = route.params;
  const { user } = useContext(AuthContext);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [tags, setTags] = useState("");
  const [location, setLocation] = useState();
  const uploadVideo = async () => {
    if (tags === "" || text === "") {
      Toast.show({
        type: "error",
        text1: "Fill in all fields",
      });
    } else {
      setLoading(true);
      const formData = new FormData();
      formData.append("video", {
        uri: route.params.uri,
        type: "video/mp4",
        name: "video",
      });
      formData.append("tags", tags);
      const response = await fetch(`${url}/videos/upload?desc=${text}`, {
        method: "POST",
        body: formData,
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${user}`,
          tags: tags,
        },
      });
      if (response.ok) {
        Toast.show({
          type: "success",
          text1: "Our Team is reviewing video for Obsene Content",
        });
        navigation.navigate("TabNavigator");
      } else {
        console.log("error");
      }
    }
  };
  if (loading) {
    return <Loader />;
  }
  return (
    <SafeAreaView style={{ backgroundColor: "white", flex: 1 }}>
      <View style={styles.uppertab}>
        <View style={styles.innertab}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Ionicons name="chevron-back" color={"black"} size={30} />
          </TouchableOpacity>
          <View>
            <Text
              style={{
                fontSize: 20,
                fontWeight: "bold",
                color: "black",
              }}
            >
              Post
            </Text>
          </View>
          <View>
            <Ionicons name="chevron-back" color={"white"} size={30} />
          </View>
        </View>
      </View>
      <ScrollView>
        <View style={{ flex: 1, marginTop: 10, height: 200 }}>
          <VideoPlayer
            source={route.params.uri}
            fullscreenIcon={false}
            recorded={true}
          />
        </View>
        <View style={{ width: "90%", alignSelf: "center" }}>
          <View
            style={{
              height: 100,
              backgroundColor: "#F5F5F5",
              borderRadius: 10,
              marginTop: 10,
            }}
          >
            <TextInput
              style={{ padding: 10, color: "darkgray" }}
              placeholder={"Describe your post"}
              placeholderTextColor="darkgray"
              multiline={true}
              numberOfLines={10}
              value={text}
              onChangeText={setText}
            />
          </View>
          <View
            style={{
              height: 50,
              backgroundColor: "#F5F5F5",
              borderRadius: 10,
              marginTop: 10,
            }}
          >
            <TextInput
              style={{ padding: 10, color: "darkgray" }}
              placeholder={"Enter Tags"}
              placeholderTextColor="darkgray"
              value={tags}
              onChangeText={setTags}
            />
          </View>
          <TouchableOpacity style={{
              backgroundColor: "#F5F5F5",
              borderRadius: 10,
              marginTop: 10,
              flexDirection:'row',
              padding:10,
              justifyContent:'space-between'
              ,alignItems:'center'
            }}>
						<View style={{ }}>
							<Ionicons
								name='location-outline'
								color={'black'}
								size={25}
								style={{ alignSelf: 'center' }}
							/>
						</View>
						<Text style={styles.text12}>{address}</Text>
					</TouchableOpacity>
        </View>

        <View style={styles.buttonview}>
          <TouchableOpacity
            style={[styles.messagebutton, { backgroundColor: "darkgray" }]}
            onPress={() => navigation.goBack()}
          >
            <Text
              style={{
                color: "white",
                fontWeight: "bold",
                fontSize: 16,
              }}
            >
              Cancel
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.messagebutton, { backgroundColor: "#FF8216" }]}
            onPress={uploadVideo}
          >
            <Text
              style={{
                color: "white",
                fontWeight: "bold",
                fontSize: 16,
              }}
            >
              Post
            </Text>
          </TouchableOpacity>
        </View>
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
    shadowColor: "#171717",
    shadowOffset: { width: -2, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 10,
  },
  innertab: {
    width: "95%",
    height: "100%",
    flexDirection: "row",
    alignItems: "center",
    marginTop: "auto",
    justifyContent: "space-between",
  },
  buttonview: {
    flexDirection: "row",
    height: 90,
    width: "85%",
    alignSelf: "center",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 60,
  },
  messagebutton: {
    justifyContent: "center",
    alignItems: "center",
    borderRadius: 50,
    width: 140,
    height: 50,
  },
  text12: {
    color: "black",
    fontWeight: "medium",
    fontSize: 16,
  },
  button11: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "white",
    height: 50,
  },
  imgs: {
    width: "100%",
    height: "100%",
    borderRadius: 10,
  },
});
