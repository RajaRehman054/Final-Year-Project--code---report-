import { Text, View, TouchableOpacity, StyleSheet } from "react-native";
import React, { useEffect, useState, useRef } from "react";
import { Avatar } from "react-native-paper";
import url from "../../utils/url.js";
import img1 from "../../assets/img1.png";

export default function Converstion({ navigation, conversation, currentUser }) {
  const [user, setUser] = useState("");
  // get user engaged in converstion
  useEffect(() => {
    const friendId = conversation.members.find((m) => m !== currentUser._id);

    const getUser = async () => {
      try {
        const res = await fetch(`${url}/users?userId=${friendId}`, {
          headers: {
            "content-type": "application/json",
          },
        });
        const response = await res.json();
        setUser(response);
      } catch (err) {
        console.log(err);
      }
    };
    getUser();
  }, [currentUser, conversation]);

  return (
    <View style={styles.notificationbar}>
      <TouchableOpacity
        style={{ width: "100%", flexDirection: "row" }}
        onPress={() =>
          navigation.navigate("DirectMessage", {
            conversation: conversation,
            currentChatUser: user,
            currentUser: currentUser,
          })
        }
      >
        <View style={styles.iconimg}>
          <Avatar.Image
            source={
              user.picture
                ? {
                    uri: `${url}/users/picture?path=${user.picture}`,
                  }
                : img1
            }
            size={50}
          />
        </View>
        <View style={styles.notifytext}>
          <View
            style={{
              flexDirection: "row",
              justifyContent: "space-between",
              height: "50%",
              width: "100%",
              alignItems: "center",
            }}
          >
            <Text style={{ fontWeight: "bold", color: "black" }}>
              {user.username}
            </Text>
            <Text style={{ color: "darkgray", marginRight: 10 }}>just now</Text>
          </View>
          <View
            style={{
              height: "50%",
              width: "100%",
              justifyContent: "center",
            }}
          >
            <Text style={{ color: "darkgray" }}>
              I'll bring last message here
            </Text>
          </View>
        </View>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  notificationbar: {
    height: 60,
    flexDirection: "row",
    backgroundColor: "white",
    marginBottom: 5,
    marginTop: 5,
  },
  iconimg: {
    width: "20%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  notifytext: {
    width: "80%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
});
