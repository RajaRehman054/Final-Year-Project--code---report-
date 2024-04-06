import { Text, View, TouchableOpacity, StyleSheet } from "react-native";
import React, { useEffect, useState } from "react";
import url from "../../utils/url.js";
import { Avatar } from "react-native-paper";
import img1 from "../../assets/img1.png";
import TimeAgo from "react-native-timeago";

export default function JobsBar({ navigation, e, job }) {
  const [user, setUser] = useState([]);

  useEffect(() => {
    const getUser = async () => {
      try {
        const res = await fetch(`${url}/users?userId=${e.creator}`, {
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
  }, [e]);
  return (
    <View style={styles.notificationbar}>
      <View style={{ width: "100%", flexDirection: "row" }}>
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
              height: "30%",
              width: "100%",
              alignItems: "center",
            }}
          >
            <Text style={{ fontWeight: "bold", color: "black" }}>
              {user.username}
            </Text>
            <TimeAgo
              style={{ color: "darkgray", marginRight: 10 }}
              time={e.createdAt}
            />
          </View>
          <View
            style={{
              height: "70%",
              width: "100%",
              justifyContent: "center",
            }}
          >
            <Text style={{ color: "darkgray" }}>{e.details}</Text>
          </View>
        </View>
        <View style={[styles.iconimg, { paddingLeft: 10, paddingRight: 10 }]}>
          {job === "Public" ? (
            <TouchableOpacity
              style={{
                backgroundColor: "#FF8216",
                height: 30,
                width: "100%",
                alignItems: "center",
                justifyContent: "center",
                borderRadius: 5,
              }}
              onPress={() =>
                navigation.navigate("Jobdetails", {
                  details: e,
                  jobPoster: user,
                  job: job,
                })
              }
            >
              <Text style={{ color: "white" }}>Apply</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={{
                backgroundColor: "#FF8216",
                height: 30,
                width: "100%",
                alignItems: "center",
                justifyContent: "center",
                borderRadius: 5,
              }}
              onPress={() =>
                navigation.navigate("Jobdetails", {
                  details: e,
                  jobPoster: user,
                  job: job,
                })
              }
            >
              <Text style={{ color: "white" }}>Go</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  notificationbar: {
    height: 100,
    flexDirection: "row",
    backgroundColor: "white",
    marginBottom: 10,

    borderRadius: 15,
    shadowColor: "#171717",
    shadowOffset: { width: -2, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 10,
  },
  iconimg: {
    width: "20%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  notifytext: {
    width: "60%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
});
