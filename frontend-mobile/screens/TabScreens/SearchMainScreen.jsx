import React, { useState } from "react";
import {
  View,
  FlatList,
  StyleSheet,
  Text,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  Image,
  ImageBackground,
  TextInput,
} from "react-native";
import Ionicons from "react-native-vector-icons/Ionicons";

const data = [
  { id: "1", name: "doja", image: require("../../assets/img1.png") },
  { id: "2", name: "doja", image: require("../../assets/img1.png") },
  { id: "3", name: "doja", image: require("../../assets/img1.png") },
  { id: "4", name: "doja", image: require("../../assets/img1.png") },
  { id: "5", name: "doja", image: require("../../assets/img1.png") },
];

const data1 = [
  {
    id: 1,
    title: "#London",
    items: [
      { id: 1, image: require("../../assets/imgrect.png") },
      { id: 2, image: require("../../assets/imgrect.png") },
    ],
  },
  {
    id: 2,
    title: "#Colleges",
    items: [
      { id: 1, image: require("../../assets/imgrect.png") },
      { id: 2, image: require("../../assets/imgrect.png") },
    ],
  },
  {
    id: 3,
    title: "#Sunsets",
    items: [
      { id: 1, image: require("../../assets/imgrect.png") },
      { id: 2, image: require("../../assets/imgrect.png") },
    ],
  },
  
];



const SearchMainScreen = ({ navigation }) => {
  const ListFooterComponent = () => (
      <View>
        <View style={{ marginTop: 10, marginBottom: 10 }}>
          <View style={styles.heading}>
            <Text
              style={{
                fontSize: 20,
                fontWeight: "medium",
                marginLeft: 10,
                color: "black",
              }}
            >
              Top Scoopers
            </Text>
            <TouchableOpacity>
              <Text style={{ marginRight: 10, color: "#0f89da", fontSize: 12 }}>
                View all
              </Text>
            </TouchableOpacity>
          </View>
          <FlatList
            data={data}
            renderItem={renderItem1}
            keyExtractor={(item) => item.id}
            horizontal={true}
          />
        </View>

        <View style={styles.heading}>
          <Text
            style={{
              fontSize: 20,
              fontWeight: "medium",
              marginLeft: 10,
              color: "black",
            }}
          >
            Trending
          </Text>
        </View>
        <FlatList
          data={data1}
          keyExtractor={(item) => item.id.toString()}
          renderItem={renderItem2}
        />
      </View>
    
);
  const renderItem1 = ({ item }) => (
    <TouchableOpacity style={styles.scooper}>
      <ImageBackground source={item.image} style={styles.image1}>
        <View style={styles.backimage}>
          <Text style={{ color: "white" }}>{item.name}</Text>
        </View>
      </ImageBackground>
    </TouchableOpacity>
  );

  const renderItem2 = ({ item }) => (
    <View style={{ marginTop: 10, marginBottom: 10 }}>
      <View style={styles.heading}>
        <Text
          style={{
            fontSize: 18,
            fontWeight: "medium",
            marginLeft: 10,
            color: "darkgray",
          }}
        >
          {item.title}
        </Text>
        <TouchableOpacity>
          <Text style={{ marginRight: 10, color: "#0f89da", fontSize: 12 }}>
            View all
          </Text>
        </TouchableOpacity>
      </View>
      <FlatList
        data={item.items}
        horizontal
        keyExtractor={(item) => item.id.toString()}
        renderItem={({ item }) => (
          <TouchableOpacity style={styles.trending}>
            <Image source={item.image} style={styles.image} />
          </TouchableOpacity>
        )}
      />
    </View>
  );

  return (
    <SafeAreaView style={{ backgroundColor: "white", flex: 1 }}>
      <View style={styles.uppertab}>
        <View style={styles.innertab}>
          <TouchableOpacity
            style={styles.container}
            onPress={() => navigation.navigate("RecentSearch")}
          >
            <Ionicons name="search" style={styles.icon} size={20} />
            <TouchableOpacity
              onPress={() => navigation.navigate("RecentSearch")}
            >
              <TextInput
                style={styles.input}
                placeholder="Search"
                placeholderTextColor={"darkgray"}
                editable={false}
              />
            </TouchableOpacity>
          </TouchableOpacity>
        </View>
      </View>

      <FlatList 
        data={[]}
        renderItem={() => null}
        ListFooterComponent={ListFooterComponent}
        style={styles.topscooper}
      />
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
    justifyContent: "center",
  },
  topscooper: {
    width: "95%",
    alignSelf: "center",
  },
  heading: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  scooper: {
    backgroundColor: "#f9c2ff",
    height: 100,
    width: 100,
    margin: 10,
    borderRadius: 10,
    shadowColor: "#171717",
    shadowOffset: { width: -2, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 5,
  },
  trending: {
    backgroundColor: "#f9c2ff",
    height: 100,
    width: 190,
    margin: 10,
    borderRadius: 10,
    shadowColor: "#171717",
    shadowOffset: { width: -2, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 5,
  },
  image: {
    width: "100%",
    height: "100%",
    resizeMode: "cover",
    borderRadius: 10,
  },
  image1: {
    width: "100%",
    height: "100%",
    resizeMode: "cover",
    alignItems: "center",
    justifyContent: "center",
  },
  container: {
    width: "90%",
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#F5F5F5",
    elevation: 5,
    height: 40,
    borderRadius: 50,    
  },
  icon: {
    color: "darkgray",
    margin: 10,
  },
  input: {
    flex: 1,
    fontSize: 18,
    color: "darkgray",
    padding:10,
  },
  backimage: {
    height: "30%",
    width: "100%",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    alignItems: "center",
    justifyContent: "center",
    marginTop: "auto",
    borderBottomLeftRadius: 10,
    borderBottomRightRadius: 10,
  },
});

export default SearchMainScreen;
