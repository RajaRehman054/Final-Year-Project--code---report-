import React, { useState, useContext, useCallback } from "react";
import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
} from "react-native";
import Ionicons from "react-native-vector-icons/Ionicons";
import { AuthContext } from "../../auth/AuthContext";
import url from "../../utils/url.js";
import NotHiredyet from "../../components/JobComp/NotHiredyet.jsx";
import CurrentlyHired from "../../components/JobComp/CurrentlyHired.jsx";
import { UserContext } from "../../context/UserContext";
import { useFocusEffect } from "@react-navigation/native";

export default function NewsJobsScreen({ navigation }) {
  const [AppliedJobs, setAppliedJobs] = useState([]);
  const { user } = useContext(AuthContext);
  const { userData, getUser } = useContext(UserContext);

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
      getAppliedJobs();
    }, [])
  );

  return (
    <SafeAreaView style={{ backgroundColor: "white", flex: 1 }}>
      <View style={styles.uppertab}>
        <View style={styles.innertab}>
          <TouchableOpacity disabled>
            <Ionicons
              name="paper-plane-outline"
              color={"transparent"}
              size={30}
            />
          </TouchableOpacity>
          <View
            style={{
              height: 40,
              borderRadius: 10,
              width: "70%",
              elevation: 5,
              backgroundColor: "#FF8216",
            }}
          >
            {userData.user.hirer === null ? (
              <Text
                style={{
                  fontSize: 20,
                  fontWeight: "bold",
                  color: "white",
                  padding: 5,
                  alignSelf: "center",
                }}
              >
                Jobs Applied | ({AppliedJobs.length})
              </Text>
            ) : (
              <Text
                style={{
                  fontSize: 20,
                  fontWeight: "bold",
                  color: "white",
                  padding: 5,
                  alignSelf: "center",
                }}
              >
                Currently Hired
              </Text>
            )}
          </View>
          <TouchableOpacity disabled>
            <Ionicons name="earth" color={"transparent"} size={30} />
          </TouchableOpacity>
        </View>
      </View>
      {userData.user.hirer === null ? (
        <NotHiredyet navigation={navigation} />
      ) : (
        <CurrentlyHired />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  uppertab: {
    height: 70,
    backgroundColor: "white",
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
});
