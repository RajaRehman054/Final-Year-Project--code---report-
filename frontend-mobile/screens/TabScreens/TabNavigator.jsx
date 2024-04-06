import { View, TouchableWithoutFeedback } from "react-native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { useNavigation } from "@react-navigation/native";
import Ionicons from "react-native-vector-icons/Ionicons";

import HomeScreen from "./HomeScreen";
import SearchMainScreen from "./SearchMainScreen";
import NewsJobsScreen from "./NewsJobsScreen";
import ProfileHomeScreen from "./ProfileHomeScreen";
import RecentSearch from "../StackScreens/RecentSearch";
const Tab = createBottomTabNavigator();

export default function TabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarStyle: {
          height: 60,
          backgroundColor: "#F5F5F5",
        },
        headerShown: false,
        tabBarActiveTintColor: "#FF8216",
        tabBarLabelStyle: { fontSize: 12, paddingBottom: 5 },
      }}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name={"home"} color={color} size={size} />
          ),
        }}
      />
      <Tab.Screen
        name="Search"
        component={RecentSearch}
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name={"search"} color={color} size={size} />
          ),
        }}
      />
      <Tab.Screen
        name="Homesc"
        component={HomeScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name={"videocam"} color={color} size={size} />
          ),
          tabBarButton: () => <MiddleTabBarButton />,
        }}
      />
      <Tab.Screen
        name="News+"
        component={NewsJobsScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name={"newspaper"} color={color} size={size} />
          ),
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileHomeScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name={"person"} color={color} size={size} />
          ),
        }}
      />
    </Tab.Navigator>
  );
}

const MiddleTabBarButton = () => {
  const navigation = useNavigation();
  return (
    <TouchableWithoutFeedback
      onPress={() => {
        navigation.navigate("CameraRecording", { text: "" });
      }}
    >
      <View
        style={{
          top: -20,
          height: 70,
          width: 70,
          borderRadius: 50,
          backgroundColor: "#FF8216",
          justifyContent: "center",
          alignItems: "center",
          borderColor: "#000000",
          elevation: 5,
        }}
      >
        <Ionicons name="videocam" color="white" size={25} />
      </View>
    </TouchableWithoutFeedback>
  );
};
