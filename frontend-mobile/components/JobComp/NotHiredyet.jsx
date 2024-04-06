import React, { useState, useRef, useContext, useCallback } from "react";
import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  Dimensions,
  TextInput,
} from "react-native";
import Ionicons from "react-native-vector-icons/Ionicons";
import { AuthContext } from "../../auth/AuthContext";
const SCREEN_WIDTH = Dimensions.get("window").width;
import url from "../../utils/url.js";
import JobsBar from "../JobComp/JobsBar.jsx";
import { UserContext } from "../../context/UserContext.jsx";
import Loader from "../Loader";
import { useFocusEffect } from "@react-navigation/native";
import Blink from "./Blink.jsx";

export default function NotHiredyet({ navigation }) {
  const [activeScreenIndex, setActiveScreenIndex] = useState(0);
  const scrollViewRef = useRef(null);
  const [publicJobs, setPublicJobs] = useState([]);
  const [AppliedJobs, setAppliedJobs] = useState([]);
  const { user } = useContext(AuthContext);
  const { userData, getUser } = useContext(UserContext);
  const [loading, setLoading] = useState(false);

  const getJobs = async () => {
    try {
      const res = await fetch(`${url}/users/jobs`, {
        headers: {
          "content-type": "application/json",
          Authorization: `Bearer ${user}`,
        },
      });
      const response = await res.json();
      setPublicJobs(response);
    } catch (err) {
      console.log(err);
    }
  };

  const getAppliedJobs = async () => {
    try {
      const res = await fetch(`${url}/users/jobs/applied`, {
        headers: {
          "content-type": "application/json",
          Authorization: `Bearer ${user}`,
        },
      });
      const response = await res.json();
      setAppliedJobs(response);
    } catch (err) {
      console.log(err);
    }
  };

  useFocusEffect(
    useCallback(() => {
      getJobs();
      getAppliedJobs();
      setLoading(false);
    }, [])
  );

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

  if (loading) {
    return <Loader />;
  }

  return (
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
            Let's find Jobs for you
          </Text>
        </View>
        <View style={styles.container}>
          <Blink duration={1000}>
            <Text
              style={{
                fontSize: 16,
                fontWeight: "bold",
                color: "#FF8216",
                textAlign: "center",
              }}
            >
              Apply on current Jobs Now
            </Text>
          </Blink>
        </View>
      </View>
      <View style={styles.followercount1}>
        <TouchableOpacity
          onPress={() => handleScreenPress(0)}
          style={{
            width: "45%",
            borderBottomColor: activeScreenIndex === 0 ? "#FF8216" : "white",
            borderBottomWidth: 3,
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Text
            style={{
              color: activeScreenIndex === 0 ? "#FF8216" : "gray",
              fontSize: 16,
              fontWeight: "bold",
            }}
          >
            Public Jobs
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => handleScreenPress(1)}
          style={{
            width: "45%",
            borderBottomColor: activeScreenIndex === 1 ? "#FF8216" : "white",
            borderBottomWidth: 3,
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Text
            style={{
              color: activeScreenIndex === 1 ? "#FF8216" : "gray",
              fontSize: 16,
              fontWeight: "bold",
            }}
          >
            Applied Jobs
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        ref={scrollViewRef}
        horizontal={true}
        pagingEnabled={true}
        showsHorizontalScrollIndicator={false}
        onMomentumScrollEnd={handleScroll}
      >
        <SafeAreaView style={styles.screen1}>
          {publicJobs.length === 0 ? (
            <Text style={{ color: "black", textAlign: "center" }}>
              No New Jobs yet
            </Text>
          ) : (
            publicJobs.map((e, index) => (
              <JobsBar
                key={index}
                navigation={navigation}
                e={e}
                job={"Public"}
              />
            ))
          )}
        </SafeAreaView>
        <SafeAreaView style={styles.screen1}>
          {AppliedJobs.length === 0 ? (
            <Text style={{ color: "black", textAlign: "center" }}>
              No jobs applied yet
            </Text>
          ) : (
            AppliedJobs.map((e, index) => (
              <JobsBar
                key={index}
                navigation={navigation}
                e={e}
                job={"Applied"}
              />
            ))
          )}
        </SafeAreaView>
      </ScrollView>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  imageicon: {
    alignItems: "center",
    justifyContent: "space-around",
    height: 160,
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
    width: "85%",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F5F5F5",
    elevation: 5,
    height: 40,
    borderRadius: 50,
    flexDirection: "row",
  },
  input: {
    fontSize: 18,
    color: "darkgray",
    padding: 10,
  },
  icon: {
    color: "darkgray",
    margin: 8,
  },
  screen1: {
    backgroundColor: "white",
    flex: 1,
    width: SCREEN_WIDTH,
    padding: 15,
  },
});
