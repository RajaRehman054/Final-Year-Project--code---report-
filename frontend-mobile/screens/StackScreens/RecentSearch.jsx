import React, { useState, useContext } from "react";
import {
  View,
  StyleSheet,
  Text,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  TextInput,
  Image,
} from "react-native";
import Ionicons from "react-native-vector-icons/Ionicons";
import { Picker } from "@react-native-picker/picker";
import { ActivityIndicator, MD2Colors } from "react-native-paper";
import SearchVideos from "../../components/SearchComp/SearchVideos";
import Loader from "../../components/Loader";
const img1 = require("../../assets/noresults.png");
import { AuthContext } from "../../auth/AuthContext";

const RecentSearch = ({ navigation }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedValue, setSelectedValue] = useState("Name");
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isSearchPressed, setIsSearchPressed] = useState(false);
  const { user } = useContext(AuthContext);

  const handleSearch = (text) => {
    setSearchQuery(text);
    setIsTyping(text !== "");
  };
  const handleValueChange = (itemValue, itemIndex) => {
    setSelectedValue(itemValue);
  };

  const handleSearchPress = async () => {
    setIsTyping(false);
    setIsSearchPressed(true);
    setLoading(true);
    searchVideosApi(selectedValue, searchQuery);
    setLoading(false);
  };

  //TODO: search Videos
  const searchVideosApi = async (type, value) => {
    try {
      const res = await fetch(`${url}/api/videos?type=${type}&value=${value}`, {
        headers: {
          "content-type": "application/json",
          Authorization: `Bearer ${user}`,
        },
      });
      const response = await res.json();
      setVideos(response);
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <SafeAreaView style={{ backgroundColor: "white", flex: 1 }}>
      <View style={styles.uppertab}>
        <View style={styles.innertab}>
          <TouchableOpacity onPress={() => navigation.goBack()} disabled>
            <Ionicons name="chevron-back" color={"transparent"} size={30} />
          </TouchableOpacity>
          <View>
            <Text
              style={{
                fontSize: 20,
                fontWeight: "medium",
                color: "black",
              }}
            >
              Search
            </Text>
          </View>
          <TouchableOpacity disabled>
            <Ionicons name="chevron-back" color={"transparent"} size={30} />
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView style={styles.topscooper}>
        <View style={styles.imageicon}>
          <View style={styles.container}>
            <Ionicons name="search" style={styles.icon} size={20} />
            <TouchableOpacity>
              <TextInput
                style={styles.input}
                placeholder="Search"
                placeholderTextColor="#FF8216"
                onChangeText={handleSearch} // Use handleSearch directly
                value={searchQuery}
              />
            </TouchableOpacity>
          </View>
          <View
            style={{
              flexDirection: "row",
              height: 55,
              width: "90%",
              justifyContent: "space-between",
            }}
          >
            <View
              style={{
                width: "48%",
              }}
            >
              <Picker
                selectedValue={selectedValue}
                onValueChange={handleValueChange}
                style={styles.picker}
                dropdownIconRippleColor={"white"}
                dropdownIconColor={"white"}
              >
                <Picker.Item label="Name" value="username" />
                <Picker.Item label="Tag" value="tags" />
                <Picker.Item label="Description" value="description" />
              </Picker>
            </View>
            <TouchableOpacity
              style={{
                backgroundColor: "white",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                width: "48%",
              }}
              disabled={searchQuery === "" ? true : false}
              onPress={() => handleSearchPress()}
            >
              <Text style={{ color: "#FF8216", fontWeight: "bold" }}>
                Search
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        {/*indicator  */}
        {isTyping && (
          <View
            style={{
              width: "100%",
              marginTop: 20,
            }}
          >
            <ActivityIndicator
              animating={true}
              size={30}
              color={MD2Colors.orange800}
            />
          </View>
        )}

        {/*results   */}
        {isSearchPressed &&
          (loading ? (
            <View
              style={{
                width: "100%",
                height: 100,
                marginTop: 20,
              }}
            >
              <Loader />
            </View>
          ) : videos.length === 0 ? (
            <View
              style={{
                width: "100%",
                marginTop: 20,
              }}
            >
              <View style={styles.followercount1}>
                <View
                  style={{
                    width: "50%",
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
                    Search Results
                  </Text>
                </View>
              </View>
              <Image
                source={img1}
                style={{
                  resizeMode: "cover",
                  width: "100%",
                }}
              />
            </View>
          ) : (
            <View
              style={{
                width: "100%",
                marginTop: 20,
              }}
            >
              <View style={styles.followercount1}>
                <View
                  style={{
                    width: "50%",
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
                    Search Results
                  </Text>
                </View>
              </View>

              <SearchVideos video={videos} navigation={navigation} />
            </View>
          ))}
      </ScrollView>
    </SafeAreaView>
  );
};

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
  topscooper: {
    width: "100%",
    padding: 10,
  },
  imageicon: {
    alignItems: "center",
    backgroundColor: "#FF8216",
    width: "100%",
    alignSelf: "center",
    borderRadius: 20,
    elevation: 10,
    height: 130,
    justifyContent: "space-evenly",
  },
  container: {
    width: "95%",
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#F5F5F5",
    alignSelf: "center",
    elevation: 5,
    height: 40,
    borderRadius: 50,
    marginBottom: 10,
  },
  icon: {
    color: "#FF8216",
    margin: 8,
  },
  input: {
    width: 250,
    fontSize: 18,
    color: "#FF8216",
    padding: 10,
  },
  picker: {
    width: "100%",
    backgroundColor: "red",
    borderRadius: 10,
  },
  followercount1: {
    flexDirection: "row",
    height: 45,
    width: "85%",
    alignSelf: "center",
    justifyContent: "space-around",
    marginBottom: 10,
  },
});

export default RecentSearch;
