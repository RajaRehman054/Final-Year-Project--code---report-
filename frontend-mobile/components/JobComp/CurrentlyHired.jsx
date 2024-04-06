import React, { useState, useContext, useCallback } from "react";
import { Text, View, StyleSheet, ScrollView } from "react-native";
import { Avatar } from "react-native-paper";
const img1 = require("../../assets/img1.png");
import TimeAgo from "react-native-timeago";
import { AuthContext } from "../../auth/AuthContext";
import url from "../../utils/url.js";
import { UserContext } from "../../context/UserContext.jsx";
import Loader from "../Loader";
import { useFocusEffect } from "@react-navigation/native";

export default function CurrentlyHired() {
  const { userData, getUser } = useContext(UserContext);
  const [hirer, setHirer] = useState([]);
  const { user } = useContext(AuthContext);
  const [hiredJob, setHiredJob] = useState([]);
  const [loading, setLoading] = useState(false);

  const getHirer = async () => {
    try {
      const res = await fetch(`${url}/users?userId=${userData.user.hirer}`, {
        headers: {
          "content-type": "application/json",
        },
      });
      const response = await res.json();
      setHirer(response);
    } catch (err) {
      console.log(err);
    }
  };

  const getHiredJob = async () => {
    try {
      const res = await fetch(`${url}/users/jobs/hiredJob`, {
        headers: {
          "content-type": "application/json",
          Authorization: `Bearer ${user}`,
        },
      });
      const response = await res.json();
      setHiredJob(response[0]);
    } catch (err) {
      console.log(err);
    }
  };

  useFocusEffect(
    useCallback(() => {
      setLoading(true);
      getHirer();
      getHiredJob();
      setLoading(false);
    }, [])
  );

  return (
    <>
      {loading ? (
        <Loader />
      ) : (
        <ScrollView>
          <View style={styles.imageicon}>
            <View style={{ width: "85%" }}>
              <Text
                style={{
                  fontSize: 23,
                  fontWeight: "bold",
                  color: "white",
                  margin: 5,
                }}
              >
                Hello! {userData.user.username}
              </Text>
              <Text
                style={{
                  fontSize: 16,
                  fontWeight: "bold",
                  color: "white",
                  margin: 5,
                }}
              >
                You are currently hired by {hirer.username}
              </Text>
            </View>

            <View style={styles.container}>
              <Text
                style={{
                  fontSize: 16,
                  fontWeight: "bold",
                  color: "red",
                  margin: 5,
                }}
              >
                Note
              </Text>
              <Text
                style={{
                  fontSize: 16,
                  color: "black",
                  margin: 5,
                }}
              >
                All videos taken by you during hiring period will belong to{" "}
                {hirer.username}
              </Text>
            </View>
          </View>
          <View style={styles.followercount1}>
            <View
              style={{
                width: "45%",
                borderBottomColor: "#FF8216",
                borderBottomWidth: 3,
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Text
                style={{
                  color: "#FF8216",
                  fontSize: 16,
                  fontWeight: "bold",
                }}
              >
                Job details
              </Text>
            </View>
          </View>
          <View style={styles.screen1}>
            <View style={styles.infotext}>
              <View style={styles.iconimg}>
                <Avatar.Image
                  source={
                    hirer.picture
                      ? {
                          uri: `${url}/users/picture?path=${hirer.picture}`,
                        }
                      : img1
                  }
                  size={50}
                />
              </View>
              <View style={styles.notifytext}>
                <View>
                  <Text style={{ color: "black" }}>
                    <Text
                      style={{
                        fontWeight: "bold",
                        color: "black",
                        fontSize: 16,
                      }}
                    >
                      {hirer.username}
                    </Text>
                    {"   "}@{hirer.username}
                  </Text>
                  <TimeAgo
                    style={{ color: "black" }}
                    time={hiredJob.createdAt}
                  />
                </View>
              </View>
            </View>
            <View>
              <Text
                style={{
                  color: "black",
                  justifyContent: "center",
                  alignItems: "center",
                  fontWeight: "bold",
                  fontSize: 20,
                }}
              >
                Description:
              </Text>
              <Text
                style={{
                  color: "black",
                  justifyContent: "center",
                  alignItems: "center",
                  marginVertical: 10,
                }}
              >
                {hiredJob.details}
              </Text>
            </View>
            <View>
              <View style={{ marginTop: 10, flexDirection: "row" }}>
                <Text style={{ color: "black", fontWeight: "bold" }}>
                  Price Offer
                </Text>

                <Text style={{ color: "black", marginLeft: 20 }}>
                  {hiredJob.price} $
                </Text>
              </View>
              <View style={{ marginTop: 10, flexDirection: "row" }}>
                <Text style={{ color: "black", fontWeight: "bold" }}>
                  Expiry Date
                </Text>
                <Text style={{ color: "black", marginLeft: 20 }}>
                  {hiredJob.expiry_date}
                </Text>
              </View>
            </View>
          </View>
        </ScrollView>
      )}
    </>
  );
}

const styles = StyleSheet.create({
  imageicon: {
    alignItems: "center",
    padding: 15,
    marginTop: 10,
    backgroundColor: "#FF8216",
    width: "92%",
    alignSelf: "center",
    borderRadius: 20,
    elevation: 10,
  },
  followercount1: {
    flexDirection: "row",
    height: 45,
    width: "100%",
    alignSelf: "center",
    justifyContent: "space-around",
    marginTop: 20,
    marginBottom: 10,
  },
  container: {
    width: "90%",
    alignItems: "center",
    backgroundColor: "#F5F5F5",
    elevation: 5,
    borderRadius: 10,
    padding: 5,
  },
  screen1: {
    backgroundColor: "white",
    flex: 1,
    width: "92%",
    alignSelf: "center",
  },
  iconimg: {
    width: "20%",
    height: "100%",
    justifyContent: "center",
  },
  notifytext: {
    width: "80%",
    height: "100%",
    justifyContent: "center",
  },
  infotext: {
    width: "100%",
    height: 80,
    alignSelf: "center",
    flexDirection: "row",
  },
});
