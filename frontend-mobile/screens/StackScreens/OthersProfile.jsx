import { useState, useEffect, useContext } from "react";
import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  FlatList,
} from "react-native";

import Ionicons from "react-native-vector-icons/Ionicons";
import { Avatar } from "react-native-paper";
import img1 from "../../assets/img1.png";
import { UserContext } from "../../context/UserContext";
import { AuthContext } from "../../auth/AuthContext";
import OtherProfileVideo from "../../components/ProfileHomeComp/OtherProfileVideo";

export default function OthersProfile({ navigation, route }) {
  const { user1 } = route.params;
  const { userData, getUser } = useContext(UserContext);
  const { user } = useContext(AuthContext);
  const [follower, setFollower] = useState(false);
  const fetchFollower = () => {
    if (userData.user.following.length !== 0) {
      const check = userData.user.following.filter(
        (elem) => elem.user._id === user1.owner._id
      );
      if (check.length > 0) {
        setFollower(true);
      }
    }
  };

  useEffect(() => {
    fetchFollower();
  }, []);

  const handleFollow = async () => {
    if (!follower) {
      await fetch(`${url}/users/follower/add/${user1.owner._id}`, {
        method: "patch",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${user}`,
        },
      });
      setFollower(true);
    } else {
      await fetch(`${url}/users/follower/remove/${user1.owner._id}`, {
        method: "patch",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${user}`,
        },
      });
      setFollower(false);
    }
  };

  const ListFooterComponent = () => (
    <View>
      <View style={styles.imageicon}>
        <Avatar.Image
          source={
            user1.owner.picture
              ? {
                  uri: `${url}/users/picture?path=${user1.owner.picture}`,
                }
              : img1
          }
          size={110}
        />
        <Text
          style={{
            fontSize: 18,
            fontWeight: "semibold",
            color: "darkgray",
          }}
        >
          @{user1.owner.username}
        </Text>
      </View>

      <View style={styles.followercount}>
        <View style={{ alignItems: "center", width: "30%" }}>
          <Text style={styles.followtext}>{user1.owner.following.length}</Text>
          <Text style={{ color: "darkgray" }}>Following</Text>
        </View>

        <View style={{ alignItems: "center", width: "30%" }}>
          <Text style={styles.followtext}>{user1.owner.followers.length}</Text>
          <Text style={{ color: "darkgray" }}>Followers</Text>
        </View>

        <View style={{ alignItems: "center", width: "30%" }}>
          <Text style={styles.followtext}>{user1.owner.sales}</Text>
          <Text style={{ color: "darkgray" }}>Sales</Text>
        </View>
      </View>
      <View style={styles.followercount2}>
        <TouchableOpacity
          style={[
            styles.messagebutton,
            { backgroundColor: follower ? "red" : "#FF8216" },
          ]}
          onPress={handleFollow}
        >
          <Text
            style={{
              color: "white",
              fontWeight: "bold",
            }}
          >
            {follower ? "Following" : "Follow"}
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.followercount1}>
        <View
          style={{
            width: "50%",
            borderBottomColor: "#FF8216",
            borderBottomWidth: 3,
            alignItems: "center",
          }}
        >
          <Ionicons name={"list"} color={"#FF8216"} size={25} />
        </View>
      </View>

      <OtherProfileVideo navigation={navigation} owner={user1.owner} />
    </View>
  );
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
                fontWeight: "medium",
                color: "black",
              }}
            >
              {user1.owner.username}
            </Text>
          </View>
          <TouchableOpacity disabled>
            <Ionicons
              name="ellipsis-horizontal"
              color={"transparent"}
              size={30}
            />
          </TouchableOpacity>
        </View>
      </View>

      <FlatList
        data={[]}
        renderItem={() => null}
        ListFooterComponent={ListFooterComponent}
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
    marginBottom: 10,
  },
  followercount2: {
    flexDirection: "row",
    height: 50,
    width: "85%",
    alignSelf: "center",
    justifyContent: "center",
    alignItems: "center",
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
  messagebutton: {
    justifyContent: "center",
    alignItems: "center",
    borderRadius: 30,
    width: 150,
    marginRight: 3,
    height: 30,
  },

  settingsbutton: {
    borderColor: "#FF8216",
    borderWidth: 2,
    justifyContent: "center",
    alignItems: "center",
    borderRadius: 30,
    width: 90,
    height: 30,
  },
});
