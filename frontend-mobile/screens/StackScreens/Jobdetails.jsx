import React, { useState, useContext, useEffect } from "react";
import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from "react-native";
import url from "../../utils/url.js";
import Ionicons from "react-native-vector-icons/Ionicons";
import { Avatar } from "react-native-paper";
const img1 = require("../../assets/img1.png");
import TimeAgo from "react-native-timeago";
import AppliedCad from "../../components/JobComp/AppliedCad.jsx";
import { AuthContext } from "../../auth/AuthContext";
import { UserContext } from "../../context/UserContext.jsx";
import Toast from "react-native-toast-message";

export default function Jobdetails({ navigation, route }) {
  const { details, jobPoster, job } = route.params;
  const { user } = useContext(AuthContext);
  const { userData, getUser } = useContext(UserContext);
  const [requestArr, setRequestArr] = useState([]);

  useEffect(() => {
    if (details.requests.length !== 0) {
      for (let i = 0; i < details.requests.length; i++) {
        const getRequests = async () => {
          try {
            const res = await fetch(
              `${url}/users?userId=${details.requests[i].user}`,
              {
                headers: {
                  "content-type": "application/json",
                },
              }
            );
            const response = await res.json();
            setRequestArr((prevArr) => [...prevArr, response]);
          } catch (err) {
            console.log(err);
          }
        };
        getRequests();
      }
    }
  }, [details]);

  // create new request
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`${url}/users/job/request/${details._id}`, {
        method: "patch",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${user}`,
        },
      });
      if (res.ok) {
        Toast.show({
          type: "success",
          text1: "Job request sent",
        });
        setRequestArr((prevArr) => [...prevArr, userData.user]);
      } else {
        Toast.show({
          type: "error",
          text1: "Job request already sent",
        });
      }
    } catch (err) {
      console.log(err);
    }
  };

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
              Job details
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
          <View style={styles.infotext}>
            <View style={styles.iconimg}>
              <Avatar.Image
                source={
                  jobPoster.picture
                    ? {
                        uri: `${url}/users/picture?path=${jobPoster.picture}`,
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
                    {jobPoster.username}
                  </Text>
                  {"   "}@{jobPoster.username}
                </Text>
                <TimeAgo style={{ color: "white" }} time={details.createdAt} />
              </View>
            </View>
          </View>
        </View>
        <View
          style={{
            width: "92%",
            alignSelf: "center",
            marginTop: 20,
          }}
        >
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
              {details.details}
            </Text>
            {job === "Applied" ? (
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
                  Note:
                </Text>
                <Text
                  style={{
                    color: "black",
                    justifyContent: "center",
                    alignItems: "center",
                    marginVertical: 10,
                  }}
                >
                  You will be notified once your job request is accepted
                </Text>
              </View>
            ) : (
              <Text></Text>
            )}
          </View>
          <View
            style={{
              height: 50,
              backgroundColor: "red",
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
              Price Offer = {details.price} $
            </Text>
          </View>
          <TouchableOpacity
            style={{
              height: 50,
              backgroundColor: "#FF8216",
              borderRadius: 10,
              justifyContent: "center",
              alignItems: "center",
              marginVertical: 10,
            }}
            onPress={handleSubmit}
            disabled={job === "Applied" ? true : false}
          >
            <Text
              style={{
                fontWeight: "bold",
                color: "white",
                fontSize: 20,
              }}
            >
              {job === "Applied" ? "Accept request pending..." : "Send Request"}
            </Text>
          </TouchableOpacity>
        </View>
        <View
          style={{
            backgroundColor: "#d9d9d9",
            height: 34,
            justifyContent: "center",
            marginTop: 10,
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
              Pending Request List
            </Text>
          </View>
        </View>
        {requestArr.length === 0 ? (
          <View
            style={{ width: "100%", alignSelf: "center", alignItems: "center" }}
          >
            <Text
              style={{
                color: "black",
                justifyContent: "center",
                alignItems: "center",
                fontWeight: "bold",
                fontSize: 16,
                marginTop: 10,
              }}
            >
              No requests yet
            </Text>
          </View>
        ) : (
          <View
            style={{ width: "100%", alignSelf: "center", alignItems: "center" }}
          >
            <Text
              style={{
                color: "black",
                justifyContent: "center",
                alignItems: "center",
                fontWeight: "bold",
                fontSize: 16,
                marginTop: 10,
              }}
            >
              No of requests on job = {requestArr.length}
            </Text>
            {requestArr.map((request, index) => (
              <AppliedCad key={index} data={request} />
            ))}
          </View>
        )}
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
